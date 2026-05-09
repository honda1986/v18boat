# -*- coding: utf-8 -*-
"""
v18.0 全艇スコア解析アプリ（Pure AI・完全データ主導型・回収率重視版）
"""

import re
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb

# ============================================================
# 基礎設定
# ============================================================
JST = timezone(timedelta(hours=+9), 'JST')

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
req_session = requests.Session()
req_session.headers.update(UA)
adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=3)
req_session.mount('https://', adapter)

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}

# ============================================================
# データ構造（Kyotei24の全指標を網羅）
# ============================================================
@dataclass
class Racer:
    name: str
    age: int
    cls_val: int  # A1:4, A2:3, B1:2, B2:1
    weight: int
    f_count: int
    avg_st: float
    n_win: float  # 全国勝率
    n_2ren: float # 全国2連率
    l_win: float  # 当地勝率
    l_2ren: float # 当地2連率
    m_2ren: float # モーター2連率
    b_2ren: float # ボート2連率

# ============================================================
# AI予測・ランキング・買い目生成
# ============================================================
@st.cache_resource
def load_lgb_model(filename: str):
    try: return lgb.Booster(model_file=filename)
    except: return None

# 新しい次元に対応するためファイル名をv18専用に変更
lgb_model   = load_lgb_model('lgb_score_v18.txt')
prob1_model = load_lgb_model('lgb_p1_v18.txt')
prob2_model = load_lgb_model('lgb_p2_v18.txt')
prob3_model = load_lgb_model('lgb_p3_v18.txt')

def get_lgb_features(r: Racer, lane: int, venue: str) -> list:
    NAME_TO_JCD = {v: k for k, v in JCD_NAME.items()}
    jcd = NAME_TO_JCD.get(venue, 1)
    # 人間の偏見を排除し、全13項目の純粋な数値をAIに流し込む
    return [
        float(jcd), float(lane), float(r.cls_val), float(r.age), float(r.weight),
        float(r.f_count), float(r.avg_st), float(r.n_win), float(r.n_2ren),
        float(r.l_win), float(r.l_2ren), float(r.m_2ren), float(r.b_2ren)
    ]

def rank_all(racers: List[Racer], venue: str) -> Tuple[List[Dict], Optional[float]]:
    out = []
    for i, r in enumerate(racers):
        lane = i + 1
        features = get_lgb_features(r, lane, venue)
        
        ai_score, p1, p2, p3 = 0.0, 0.0, 0.0, 0.0
        
        if lgb_model:   ai_score = round(lgb_model.predict([features])[0] * 10, 2)
        if prob1_model: p1 = round(prob1_model.predict([features])[0] * 100, 1)
        if prob2_model: p2 = round(prob2_model.predict([features])[0] * 100, 1)
        if prob3_model: p3 = round(prob3_model.predict([features])[0] * 100, 1)

        out.append({
            "lane": lane, "racer": r, "score": ai_score,
            "1着率": p1, "2着率": p2, "3着率": p3
        })
        
    out.sort(key=lambda x: x["score"], reverse=True)
    lane1_prob = next((x["1着率"] for x in out if x["lane"] == 1), None)
    return out, lane1_prob

def make_bets(ranked: List[Dict], strategy: str = "roi") -> List[str]:
    if len(ranked) < 4: return []
    
    # 確率またはスコアでソート
    def get_rate(x, key):
        return x.get(key) if x.get(key) > 0 else x["score"]

    lanes_by_1 = [x["lane"] for x in sorted(ranked, key=lambda x: get_rate(x, "1着率"), reverse=True)]
    lanes_by_2 = [x["lane"] for x in sorted(ranked, key=lambda x: get_rate(x, "2着率"), reverse=True)]
    lanes_by_3 = [x["lane"] for x in sorted(ranked, key=lambda x: get_rate(x, "3着率"), reverse=True)]

    l1 = lanes_by_1[0] 
    c2 = [l for l in lanes_by_2 if l != l1]
    c3 = [l for l in lanes_by_3 if l != l1]
    
    raw = []
    if strategy == "safe": 
        # 本命2点
        for s in c2[:2]:
            t_cands = [t for t in c3 if t != s]
            if t_cands: raw.append(f"{l1}-{s}-{t_cands[0]}")
    elif strategy == "standard": 
        # 標準4点
        for s in c2[:2]:
            for t in [t for t in c3 if t != s][:2]: raw.append(f"{l1}-{s}-{t}")
    elif strategy == "roi": 
        # 回収率重視（中穴カバーの6点）
        for s in c2[:3]:
            for t in [t for t in c3 if t != s][:2]: raw.append(f"{l1}-{s}-{t}")
    else: # wide
        for s in c2[:3]:
            for t in [t for t in c3 if t != s][:3]: raw.append(f"{l1}-{s}-{t}")

    unique_bets = []
    for b in raw:
        if b not in unique_bets: unique_bets.append(b)
            
    return unique_bets

def strategy_label(strategy: str) -> str:
    return {"safe": "安全(2点)", "standard": "標準(4点)", "roi": "回収率重視(6点)", "wide": "広め(9点)"}.get(strategy, strategy)

# ============================================================
# スクレイピング関数 (Kyotei24専用・全情報抽出)
# ============================================================
from bs4 import BeautifulSoup

def fetch_kyotei24_data(jcd: int, rno: int, dstr: str):
    url = f"https://info.kyotei.fun/info-{dstr}-{jcd:02d}-{rno}.html"
    try:
        r = req_session.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        html = r.text if r.status_code == 200 else ""
    except:
        return None

    if not html or "出走表" not in html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # 1. 順位（結果）の抽出
    lane_to_rank = {}
    jyuni_divs = soup.find_all('div', class_='jyuni')
    has_result = False
    if len(jyuni_divs) >= 6:
        for i in range(6):
            txt = jyuni_divs[i].get_text(strip=True)
            if txt.isdigit():
                lane_to_rank[i+1] = txt
                has_result = True

    # 2. 払戻金の抽出
    payoff = 0
    if has_result:
        payoff_div = soup.find('div', class_='race_result_end_label', string=re.compile('3連単'))
        if payoff_div and payoff_div.parent:
            money_span = payoff_div.parent.find('span', class_='race_result_end_money_num')
            if money_span:
                ptxt = money_span.get_text(strip=True).replace(',', '')
                if ptxt.isdigit(): payoff = int(ptxt)

    win_combo = ""
    r1 = next((k for k, v in lane_to_rank.items() if str(v) == '1'), None)
    r2 = next((k for k, v in lane_to_rank.items() if str(v) == '2'), None)
    r3 = next((k for k, v in lane_to_rank.items() if str(v) == '3'), None)
    if r1 and r2 and r3: win_combo = f"{r1}-{r2}-{r3}"

    # 3. 選手全データの抽出
    rd = [{"name": f"選手{i+1}", "age": 30, "cls": 1, "weight": 50, "f": 0, 
           "st": 0.17, "nw": 0.0, "n2": 0.0, "lw": 0.0, "l2": 0.0, "m2": 0.0, "b2": 0.0} for i in range(6)]
    
    cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

    for tr in soup.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if len(tds) >= 7:
            label = tds[0].get_text(strip=True).replace('\n', '').replace(' ', '')
            
            if "選手名" in label:
                for i in range(6):
                    txt = tds[i+1].get_text(strip=True)
                    rd[i]["name"] = re.sub(r'[\d\(\)\s]', '', txt)
                    m_age = re.search(r'\((\d+)\)', txt)
                    if m_age: rd[i]["age"] = int(m_age.group(1))
            elif label.startswith("級"):
                for i in range(6):
                    m = re.search(r'([A12B]{2})', tds[i+1].get_text(strip=True))
                    if m: rd[i]["cls"] = cls_map.get(m.group(1), 1)
            elif "選手情報" in label:
                for i in range(6):
                    m = re.search(r'(\d+)kg', tds[i+1].get_text(strip=True))
                    if m: rd[i]["weight"] = int(m.group(1))
            elif "全国" in label and "勝率" in label:
                for i in range(6):
                    txt = tds[i+1].get_text(strip=True)
                    m_2 = re.search(r'^([\d\.]+)', txt)
                    m_w = re.search(r'\(([\d\.]+)\)', txt)
                    if m_2: rd[i]["n2"] = float(m_2.group(1))/100.0 if float(m_2.group(1))>1.0 else float(m_2.group(1))
                    if m_w: rd[i]["nw"] = float(m_w.group(1))
            elif "当地" in label and "勝率" in label:
                for i in range(6):
                    txt = tds[i+1].get_text(strip=True)
                    m_2 = re.search(r'^([\d\.]+)', txt)
                    m_w = re.search(r'\(([\d\.]+)\)', txt)
                    if m_2: rd[i]["l2"] = float(m_2.group(1))/100.0 if float(m_2.group(1))>1.0 else float(m_2.group(1))
                    if m_w: rd[i]["lw"] = float(m_w.group(1))
            elif "モータ" in label and "2連率" in label:
                for i in range(6):
                    m = re.search(r'^([\d\.]+)', tds[i+1].get_text(strip=True))
                    if m: rd[i]["m2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "ボート" in label and "2連率" in label:
                for i in range(6):
                    m = re.search(r'^([\d\.]+)', tds[i+1].get_text(strip=True))
                    if m: rd[i]["b2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "平均ST" in label:
                for i in range(6):
                    try: rd[i]["st"] = float(tds[i+1].get_text(strip=True))
                    except: pass
            elif "フライング" in label:
                for i in range(6):
                    try: rd[i]["f"] = int(tds[i+1].get_text(strip=True))
                    except: pass

    if sum(x["nw"] for x in rd) == 0:
        return None

    racers = []
    for x in rd:
        racers.append(Racer(
            name=x["name"], age=x["age"], cls_val=x["cls"], weight=x["weight"], f_count=x["f"],
            avg_st=x["st"], n_win=x["nw"], n_2ren=x["n2"], l_win=x["lw"], l_2ren=x["l2"],
            m_2ren=x["m2"], b_2ren=x["b2"]
        ))
        
    return racers, lane_to_rank, win_combo, payoff, has_result

# ============================================================
# メインUI
# ============================================================
st.set_page_config(page_title="v18.0 Pure AI", layout="wide")
st.title("🚤 v18.0 全艇スコア解析 (Pure AI)")
st.caption("完全データ主導型AI / 回収率重視ロジック搭載")

if not lgb_model:
    st.warning("⚠️ 学習済みのAIモデルがありません。AIのスコアはすべて0で計算されます。タブ2からデータを収集してモデルを作成してください。")

tab1, tab2 = st.tabs(["🔍 1レース解析", "📊 バックテスト & データ収集"])

# ----------------------------------------------------
# タブ1: 1レース解析
# ----------------------------------------------------
with tab1:
    st.subheader("🔍 1レース解析")
    col1, col2 = st.columns(2)
    with col1:
        d_input = st.date_input("日付", value=datetime.now(JST).date())
    with col2:
        v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
        
    r_idx = st.selectbox("レース", options=list(range(1, 13)))
    
    if st.button("🔍 解析開始", type="primary", use_container_width=True):
        dstr = d_input.strftime("%Y%m%d")
        res = fetch_kyotei24_data(v_idx, r_idx, dstr)
        if res:
            racers, _, _, _, _ = res
            ranked, lane1_prob = rank_all(racers, JCD_NAME[v_idx])
            
            st.success("解析完了！")
            
            if lane1_prob is not None and lane1_prob > 0:
                st.markdown(f"### 🎯 1号艇の逃げ切り確率: **<span style='color:red;'>{lane1_prob}%</span>**", unsafe_allow_html=True)
                
            df_disp = []
            for item in ranked:
                racer = item["racer"]
                df_disp.append({
                    "枠": item["lane"],
                    "選手名": racer.name,
                    "AIスコア": item["score"],
                    "1着率(%)": item["1着率"],
                    "2着率(%)": item["2着率"],
                    "3着率(%)": item["3着率"],
                    "階級値": racer.cls_val,
                    "全国勝率": racer.n_win,
                    "当地勝率": racer.l_win,
                    "モータ2連": racer.m_2ren,
                    "平均ST": racer.avg_st,
                    "F数": racer.f_count
                })
            
            df_out = pd.DataFrame(df_disp).set_index("枠")
            st.dataframe(df_out, use_container_width=True)
            
            st.subheader("💡 おすすめ買い目")
            bets_roi = make_bets(ranked, "roi")
            bets_std = make_bets(ranked, "standard")
            bets_wide = make_bets(ranked, "wide")
            st.write(f"**回収率重視(6点)**: {', '.join(bets_roi) if bets_roi else 'なし'}")
            st.write(f"**標準(4点)**: {', '.join(bets_std) if bets_std else 'なし'}")
            st.write(f"**広め(9点)**: {', '.join(bets_wide) if bets_wide else 'なし'}")
        else:
            st.error("出走表が取得できませんでした。")

# ----------------------------------------------------
# タブ2: バックテスト & データ収集
# ----------------------------------------------------
with tab2:
    st.subheader("📊 期間バックテスト & Colab学習データ出力")
    
    col1, col2 = st.columns(2)
    with col1:
        bt_start = st.date_input("開始日 ", value=datetime.now(JST).date())
    with col2:
        bt_end = st.date_input("終了日 ", value=datetime.now(JST).date())
        
    bt_venue_idx = st.selectbox("場を指定", options=[0] + list(JCD_NAME.keys()), format_func=lambda x: "全国（すべて）" if x==0 else JCD_NAME[x])
    bt_strategy = st.radio("買い目戦略", options=["safe", "standard", "roi", "wide"], format_func=strategy_label, horizontal=True, index=2)

    if st.button("📊 バックテスト実行", type="primary", use_container_width=True):
        days = [(bt_start + timedelta(days=i)).strftime("%Y%m%d") for i in range((bt_end - bt_start).days + 1)]
        matches = []
        prog = st.progress(0.0)
        
        tasks = []
        jcds_to_check = [bt_venue_idx] if bt_venue_idx != 0 else list(range(1, 25))
        for dstr in days:
            for j in jcds_to_check:
                for r in range(1, 13):
                    tasks.append((dstr, j, r))
                    
        st.write(f"全 {len(tasks)} レースを解析中...")
        
        def analyze_race(d, j, r):
            res = fetch_kyotei24_data(j, r, d)
            if not res: return None
            
            racers, lane_to_rank, actual_result, payoff, has_result = res
            venue_name = JCD_NAME.get(j, "不明")
            ranked, lane1_prob = rank_all(racers, venue_name)
            bets = make_bets(ranked, strategy=bt_strategy)
            
            top_score = ranked[0]["score"]
            
            # 結果が出ているかどうかの判定
            if not has_result:
                hit_str = "⏳"
                payoff_disp = "-"
                train_rows = []
                actual_result = "結果待ち"
                hit_amount = 0
            else:
                hit = actual_result in bets
                hit_str = "🎯" if hit else "❌"
                payoff_disp = payoff
                hit_amount = payoff if hit else 0
                
                # 学習データ生成
                train_rows = []
                for i, r_obj in enumerate(racers):
                    lane = i + 1
                    rank_str = str(lane_to_rank.get(lane, '6'))
                    score = { '1': 1.0, '2': 0.8, '3': 0.6, '4': 0.4, '5': 0.2 }.get(rank_str, 0.0)
                    train_rows.append({
                        "場": j, "枠番": lane, 
                        "級": r_obj.cls_val, "年齢": r_obj.age, "体重": r_obj.weight,
                        "F数": r_obj.f_count, "平均ST": r_obj.avg_st, 
                        "全国勝率": r_obj.n_win, "全国2連": r_obj.n_2ren,
                        "当地勝率": r_obj.l_win, "当地2連": r_obj.l_2ren,
                        "モータ2連": r_obj.m_2ren, "ボート2連": r_obj.b_2ren,
                        "target_score": score, 
                        "target_1": 1 if rank_str == '1' else 0, 
                        "target_2": 1 if rank_str == '2' else 0, 
                        "target_3": 1 if rank_str == '3' else 0
                    })
            
            return {
                "日付": d,
                "場": venue_name,
                "R": r,
                "買い目": ", ".join(bets),
                "点数": len(bets),
                "結果": actual_result,
                "的中": hit_str,
                "払戻金": payoff_disp,
                "_hit_amount": hit_amount,
                "AIトップ": top_score,
                "_train_rows": train_rows 
            }
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            future_to_task = {executor.submit(analyze_race, d, j, r): (d, j, r) for d, j, r in tasks}
            done_count = 0
            for future in concurrent.futures.as_completed(future_to_task):
                done_count += 1
                prog.progress(done_count / len(tasks) if len(tasks) > 0 else 1.0)
                res = future.result()
                if res:
                    matches.append(res)
                    
        # 日付・場・R順にソート
        matches.sort(key=lambda x: (x["日付"], x["場"], x["R"]))

        if matches:
            df_bt = pd.DataFrame(matches)
            
            # 完了しているレースのみで回収率を計算
            finished_races = [m for m in matches if m["的中"] in ["🎯", "❌"]]
            hits = [m for m in finished_races if m["的中"] == "🎯"]
            
            if finished_races:
                total_invest = sum(m["点数"] for m in finished_races) * 100
                total_return = sum(m["_hit_amount"] for m in finished_races)
                hit_rate = len(hits) / len(finished_races) * 100
                ret_rate = total_return / total_invest * 100 if total_invest > 0 else 0
                
                st.success(f"結果確定レース: {len(finished_races)}件 / 的中: {len(hits)}件 (的中率 {hit_rate:.1f}%)")
                st.info(f"💰 **総投資**: {total_invest:,}円 / **総回収**: {total_return:,}円 (回収率: {ret_rate:.1f}%)")
            
            disp_cols = ["日付", "場", "R", "買い目", "結果", "的中", "払戻金", "AIトップ"]
            st.dataframe(df_bt[disp_cols], use_container_width=True)
            
            # 学習データエクスポート
            all_train_data = []
            for m in matches:
                all_train_data.extend(m["_train_rows"])
                
            if all_train_data:
                st.write("---")
                st.markdown(f"### 🧠 Google Colab用 学習データ書き出し")
                st.write(f"全 {len(all_train_data)} 艇分の最新データを、AIが学習するためのCSVとしてダウンロードできます。")
                
                df_train = pd.DataFrame(all_train_data)
                csv_data = df_train.to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="📥 Colab学習用CSV (kyotei_pure_ai_data.csv) をダウンロード",
                    data=csv_data,
                    file_name="kyotei_pure_ai_data.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
        else:
            st.warning("解析できるレースがありませんでした。")
