# -*- coding: utf-8 -*-
"""
v18.5 全艇スコア解析アプリ（Pure AI・相対化ロジック＆自信度フィルター搭載）
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
from bs4 import BeautifulSoup

# ============================================================
# 基礎設定
# ============================================================
JST = timezone(timedelta(hours=+9), 'JST')
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
req_session = requests.Session()
req_session.headers.update(UA)
adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=3)
req_session.mount('https://', adapter)
req_session.mount('http://', adapter)

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}

@dataclass
class Racer:
    name: str; age: int; cls_val: int; weight: int; f_count: int; avg_st: float
    n_win: float; n_2ren: float; l_win: float; l_2ren: float; m_2ren: float; b_2ren: float

# ============================================================
# 相対データ（偏差・順位）計算ロジック
# ============================================================
def calc_relative_stats(racers: List[Racer]) -> List[Dict]:
    avg_win = sum(r.n_win for r in racers) / 6.0
    avg_motor = sum(r.m_2ren for r in racers) / 6.0
    avg_st = sum(r.avg_st for r in racers) / 6.0

    win_rates = sorted([r.n_win for r in racers], reverse=True)
    motors = sorted([r.m_2ren for r in racers], reverse=True)
    sts = sorted([r.avg_st for r in racers]) # STは小さい方が1位

    stats = []
    for r in racers:
        win_dev = round(r.n_win - avg_win, 2)
        motor_dev = round(r.m_2ren - avg_motor, 4)
        st_dev = round(avg_st - r.avg_st, 3) # 正の値なら平均より速い
        
        win_rank = win_rates.index(r.n_win) + 1
        motor_rank = motors.index(r.m_2ren) + 1
        st_rank = sts.index(r.avg_st) + 1

        stats.append({
            "win_dev": win_dev, "motor_dev": motor_dev, "st_dev": st_dev,
            "win_rank": win_rank, "motor_rank": motor_rank, "st_rank": st_rank
        })
    return stats

# ============================================================
# AI予測
# ============================================================
@st.cache_resource
def load_lgb_model(filename: str):
    try: return lgb.Booster(model_file=filename)
    except: return None

# 新次元（19項目）対応のモデル名に変更
lgb_model   = load_lgb_model('lgb_score_v18_5.txt')
prob1_model = load_lgb_model('lgb_p1_v18_5.txt')
prob2_model = load_lgb_model('lgb_p2_v18_5.txt')
prob3_model = load_lgb_model('lgb_p3_v18_5.txt')

def get_lgb_features(r: Racer, lane: int, venue: str, rel: Dict) -> list:
    jcd = {v: k for k, v in JCD_NAME.items()}.get(venue, 1)
    # 従来の13項目 ＋ 新しい相対6項目 ＝ 計19項目
    return [
        float(jcd), float(lane), float(r.cls_val), float(r.age), float(r.weight),
        float(r.f_count), float(r.avg_st), float(r.n_win), float(r.n_2ren),
        float(r.l_win), float(r.l_2ren), float(r.m_2ren), float(r.b_2ren),
        float(rel["win_dev"]), float(rel["motor_dev"]), float(rel["st_dev"]),
        float(rel["win_rank"]), float(rel["motor_rank"]), float(rel["st_rank"])
    ]

def rank_all(racers: List[Racer], venue: str) -> Tuple[List[Dict], Optional[float]]:
    out = []
    rel_stats = calc_relative_stats(racers)
    for i, r in enumerate(racers):
        lane = i + 1
        features = get_lgb_features(r, lane, venue, rel_stats[i])
        ai_score, p1, p2, p3 = 0.0, 0.0, 0.0, 0.0
        
        if lgb_model:   ai_score = round(lgb_model.predict([features])[0] * 10, 2)
        if prob1_model: p1 = round(prob1_model.predict([features])[0] * 100, 1)
        if prob2_model: p2 = round(prob2_model.predict([features])[0] * 100, 1)
        if prob3_model: p3 = round(prob3_model.predict([features])[0] * 100, 1)

        out.append({"lane": lane, "racer": r, "score": ai_score, "1着率": p1, "2着率": p2, "3着率": p3, "rel": rel_stats[i]})
        
    out.sort(key=lambda x: x["score"], reverse=True)
    lane1_prob = next((x["1着率"] for x in out if x["lane"] == 1), None)
    return out, lane1_prob

def make_bets(ranked: List[Dict], strategy: str = "roi") -> List[str]:
    if len(ranked) < 4: return []
    def get_rate(x, key): return x.get(key) if x.get(key) > 0 else x["score"]

    lanes_by_1 = [x["lane"] for x in sorted(ranked, key=lambda x: get_rate(x, "1着率"), reverse=True)]
    lanes_by_2 = [x["lane"] for x in sorted(ranked, key=lambda x: get_rate(x, "2着率"), reverse=True)]
    lanes_by_3 = [x["lane"] for x in sorted(ranked, key=lambda x: get_rate(x, "3着率"), reverse=True)]

    l1 = lanes_by_1[0] 
    c2 = [l for l in lanes_by_2 if l != l1]
    c3 = [l for l in lanes_by_3 if l != l1]
    
    raw = []
    if strategy == "safe": 
        for s in c2[:2]:
            t_cands = [t for t in c3 if t != s]
            if t_cands: raw.append(f"{l1}-{s}-{t_cands[0]}")
    elif strategy == "standard": 
        for s in c2[:2]:
            for t in [t for t in c3 if t != s][:2]: raw.append(f"{l1}-{s}-{t}")
    elif strategy == "roi": 
        for s in c2[:3]:
            for t in [t for t in c3 if t != s][:2]: raw.append(f"{l1}-{s}-{t}")
    else:
        for s in c2[:3]:
            for t in [t for t in c3 if t != s][:3]: raw.append(f"{l1}-{s}-{t}")

    unique_bets = []
    for b in raw:
        if b not in unique_bets: unique_bets.append(b)
    return unique_bets

def strategy_label(strategy: str) -> str:
    return {"safe": "安全(2点)", "standard": "標準(4点)", "roi": "回収率重視(6点)", "wide": "広め(9点)"}.get(strategy, strategy)

# ============================================================
# スクレイピング関数
# ============================================================
def fetch_kyotei24_data(jcd: int, rno: int, dstr: str):
    url = f"https://info.kyotei.fun/info-{dstr}-{jcd:02d}-{rno}.html"
    try:
        r = req_session.get(url, timeout=7)
        r.encoding = r.apparent_encoding
        html = r.text if r.status_code == 200 else ""
    except: return None
    if not html or "出走表" not in html: return None

    try: soup = BeautifulSoup(html, "lxml")
    except: soup = BeautifulSoup(html, "html.parser")

    lane_to_rank = {}
    jyuni_divs = soup.find_all('div', class_='jyuni')
    has_result = False
    if len(jyuni_divs) >= 6:
        for i in range(6):
            txt = jyuni_divs[i].get_text(strip=True)
            if txt.isdigit():
                lane_to_rank[i+1] = txt
                has_result = True

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

    rd = [{"name": f"選手{i+1}", "age": 30, "cls": 1, "weight": 50, "f": 0, "st": 0.17, "nw": 0.0, "n2": 0.0, "lw": 0.0, "l2": 0.0, "m2": 0.0, "b2": 0.0} for i in range(6)]
    cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    current_label = ""

    for tr in soup.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if not tds: continue
        if len(tds) >= 7:
            current_label = tds[0].get_text(strip=True).replace('\n', '').replace(' ', '')
            data_tds = tds[-6:]
        elif len(tds) == 6 and current_label:
            data_tds = tds
        else:
            current_label = ""
            continue
            
        for i in range(6):
            txt_raw = data_tds[i].get_text(" ", strip=True)
            txt_nospace = txt_raw.replace(' ', '').replace('　', '').replace('\n', '')
            if "選手名" in current_label:
                m_age = re.search(r'\((\d{2})\)', txt_nospace)
                if m_age: rd[i]["age"] = int(m_age.group(1))
                name_clean = re.sub(r'[\d\(\)\s]', '', txt_raw)
                if name_clean: rd[i]["name"] = name_clean
            elif "選手情報" in current_label or "支部" in current_label or "級" in current_label:
                m_cls = re.search(r'([A12B]{2})', txt_nospace)
                if m_cls: rd[i]["cls"] = cls_map.get(m_cls.group(1), 1)
                m_w = re.search(r'(\d+)kg', txt_nospace, re.IGNORECASE)
                if m_w: rd[i]["weight"] = int(m_w.group(1))
            elif "全国" in current_label and "勝率" in current_label:
                m_2 = re.search(r'^([\d\.]+)', txt_nospace)
                m_w = re.search(r'\(([\d\.]+)\)', txt_nospace)
                if m_2: rd[i]["n2"] = float(m_2.group(1))/100.0 if float(m_2.group(1))>1.0 else float(m_2.group(1))
                if m_w: rd[i]["nw"] = float(m_w.group(1))
            elif "当地" in current_label and "勝率" in current_label:
                m_2 = re.search(r'^([\d\.]+)', txt_nospace)
                m_w = re.search(r'\(([\d\.]+)\)', txt_nospace)
                if m_2: rd[i]["l2"] = float(m_2.group(1))/100.0 if float(m_2.group(1))>1.0 else float(m_2.group(1))
                if m_w: rd[i]["lw"] = float(m_w.group(1))
            elif "モータ" in current_label and "2連率" in current_label:
                m = re.search(r'^([\d\.]+)', txt_nospace)
                if m: rd[i]["m2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "ボート" in current_label and "2連率" in current_label:
                m = re.search(r'^([\d\.]+)', txt_nospace)
                if m: rd[i]["b2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "平均ST" in current_label:
                try: rd[i]["st"] = float(txt_nospace)
                except: pass
            elif "フライング" in current_label:
                try: rd[i]["f"] = int(txt_nospace)
                except: pass

    if sum(x["nw"] for x in rd) == 0: return None
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
st.set_page_config(page_title="v18.5 特徴量強化版", layout="wide")
st.title("🚤 v18.5 全艇スコア解析 (Pure AI)")
st.caption("完全データ主導型AI / 📊 相対化データ（偏差・順位）搭載モデル")

if not lgb_model:
    st.warning("⚠️ v18.5用のAIモデルがありません。AIのスコアはすべて0で計算されます。")

tab1, tab2 = st.tabs(["🔍 1レース解析", "📊 バックテスト & データ収集"])

# ----------------------------------------------------
# タブ1: 1レース解析
# ----------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1: d_input = st.date_input("日付", value=datetime.now(JST).date())
    with col2: v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
    r_idx = st.selectbox("レース", options=list(range(1, 13)))
    
    if st.button("🔍 解析開始", type="primary", use_container_width=True):
        dstr = d_input.strftime("%Y%m%d")
        res = fetch_kyotei24_data(v_idx, r_idx, dstr)
        if res:
            racers, _, _, _, _ = res
            ranked, lane1_prob = rank_all(racers, JCD_NAME[v_idx])
            
            top_prob = max([x["1着率"] for x in ranked])
            if top_prob >= 70:
                st.markdown(f"🔥 **【勝負レース推奨】** AIトップ艇の1着率: **<span style='color:red;'>{top_prob}%</span>**", unsafe_allow_html=True)
            else:
                st.markdown(f"⚠️ **【見送り推奨】** 混戦模様（トップの1着率: {top_prob}%）")
                
            df_disp = []
            for item in ranked:
                racer = item["racer"]
                rel = item["rel"]
                df_disp.append({
                    "枠": item["lane"], "選手名": racer.name, "AIスコア": item["score"], "1着率(%)": item["1着率"],
                    "勝率": racer.n_win, "勝率(偏差)": f"{rel['win_dev']:+.2f}", "勝率(順位)": f"{rel['win_rank']}位",
                    "モータ": racer.m_2ren, "ﾓｰﾀ(偏差)": f"{rel['motor_dev']:+.2f}", "ﾓｰﾀ(順位)": f"{rel['motor_rank']}位",
                    "ST": racer.avg_st, "ST(偏差)": f"{rel['st_dev']:+.3f}", "ST(順位)": f"{rel['st_rank']}位",
                })
            st.dataframe(pd.DataFrame(df_disp).set_index("枠"), use_container_width=True)
            
            st.subheader("💡 おすすめ買い目")
            st.write(f"**回収率重視(6点)**: {', '.join(make_bets(ranked, 'roi'))}")
        else:
            st.error("出走表が取得できませんでした。")

# ----------------------------------------------------
# タブ2: バックテスト & データ収集
# ----------------------------------------------------
with tab2:
    col1, col2, col3 = st.columns(3)
    with col1: bt_start = st.date_input("開始日 ", value=datetime.now(JST).date())
    with col2: bt_end = st.date_input("終了日 ", value=datetime.now(JST).date())
    with col3: bt_venue_idx = st.selectbox("場を指定", options=[0] + list(JCD_NAME.keys()), format_func=lambda x: "全国（すべて）" if x==0 else JCD_NAME[x])
    
    st.markdown("#### 🎯 抽出フィルター設定")
    confidence_filter = st.slider("【見送りライン】トップ艇の1着率が何%以上なら投票するか（0で全レース投票）", 0, 100, 0, 5)
    bt_strategy = st.radio("買い目戦略", options=["safe", "standard", "roi", "wide"], format_func=strategy_label, horizontal=True, index=2)

    if st.button("📊 バックテスト実行", type="primary", use_container_width=True):
        days = [(bt_start + timedelta(days=i)).strftime("%Y%m%d") for i in range((bt_end - bt_start).days + 1)]
        matches = []
        prog = st.progress(0.0)
        
        tasks = [(dstr, j, r) for dstr in days for j in ([bt_venue_idx] if bt_venue_idx != 0 else list(range(1, 25))) for r in range(1, 13)]
                    
        st.write(f"全 {len(tasks)} レースを解析中（⚡ 30並列モード）...")
        
        def analyze_race(d, j, r):
            res = fetch_kyotei24_data(j, r, d)
            if not res: return None
            
            racers, lane_to_rank, actual_result, payoff, has_result = res
            venue_name = JCD_NAME.get(j, "不明")
            ranked, _ = rank_all(racers, venue_name)
            
            top_prob = max([x["1着率"] for x in ranked]) if ranked else 0
            is_confident = top_prob >= confidence_filter
            bets = make_bets(ranked, strategy=bt_strategy) if is_confident else []
            
            if not has_result:
                hit_str, payoff_disp, actual_result, hit_amount = "⏳", "-", "結果待ち" if is_confident else "見送り推奨", 0
            else:
                if not is_confident:
                    hit_str, payoff_disp, actual_result, hit_amount = "ー", f"({payoff})", "見送り", 0
                else:
                    hit = actual_result in bets
                    hit_str, payoff_disp, hit_amount = ("🎯" if hit else "❌"), payoff, (payoff if hit else 0)
                
            train_rows = []
            if has_result:
                rel_stats = calc_relative_stats(racers)
                for i, r_obj in enumerate(racers):
                    lane = i + 1
                    rank_str = str(lane_to_rank.get(lane, '6'))
                    score = { '1': 1.0, '2': 0.8, '3': 0.6, '4': 0.4, '5': 0.2 }.get(rank_str, 0.0)
                    train_rows.append({
                        "場": j, "枠番": lane, "級": r_obj.cls_val, "年齢": r_obj.age, "体重": r_obj.weight,
                        "F数": r_obj.f_count, "平均ST": r_obj.avg_st, "全国勝率": r_obj.n_win, "全国2連": r_obj.n_2ren,
                        "当地勝率": r_obj.l_win, "当地2連": r_obj.l_2ren, "モータ2連": r_obj.m_2ren, "ボート2連": r_obj.b_2ren,
                        # 新規：相対データ
                        "勝率_偏差": rel_stats[i]["win_dev"], "モータ_偏差": rel_stats[i]["motor_dev"], "ST_偏差": rel_stats[i]["st_dev"],
                        "勝率_順位": rel_stats[i]["win_rank"], "モータ_順位": rel_stats[i]["motor_rank"], "ST_順位": rel_stats[i]["st_rank"],
                        "target_score": score, "target_1": 1 if rank_str == '1' else 0, 
                        "target_2": 1 if rank_str == '2' else 0, "target_3": 1 if rank_str == '3' else 0
                    })
            
            return {
                "日付": d, "場": venue_name, "R": r, "トップ勝率": f"{top_prob}%",
                "買い目": ", ".join(bets) if bets else "見", "点数": len(bets), "結果": actual_result, "的中": hit_str,
                "払戻金": payoff_disp, "_hit_amount": hit_amount, "_train_rows": train_rows 
            }
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            future_to_task = {executor.submit(analyze_race, d, j, r): (d, j, r) for d, j, r in tasks}
            done_count = 0
            for future in concurrent.futures.as_completed(future_to_task):
                done_count += 1
                prog.progress(done_count / len(tasks) if len(tasks) > 0 else 1.0)
                res = future.result()
                if res: matches.append(res)
                    
        matches.sort(key=lambda x: (x["日付"], x["場"], x["R"]))

        if matches:
            df_bt = pd.DataFrame(matches)
            bet_races = [m for m in matches if m["点数"] > 0 and m["的中"] in ["🎯", "❌"]]
            hits = [m for m in bet_races if m["的中"] == "🎯"]
            
            if bet_races:
                total_invest = sum(m["点数"] for m in bet_races) * 100
                total_return = sum(m["_hit_amount"] for m in bet_races)
                hit_rate = len(hits) / len(bet_races) * 100
                ret_rate = total_return / total_invest * 100 if total_invest > 0 else 0
                st.success(f"🔥 勝負レース: {len(bet_races)}件 (見送り: {len([m for m in matches if m['結果']=='見送り'])}件)")
                st.info(f"💰 **総投資**: {total_invest:,}円 / **総回収**: {total_return:,}円 (回収率: {ret_rate:.1f}%)")
            
            disp_cols = ["日付", "場", "R", "トップ勝率", "買い目", "結果", "的中", "払戻金"]
            st.dataframe(df_bt[disp_cols], use_container_width=True)
            
            all_train_data = []
            for m in matches: all_train_data.extend(m["_train_rows"])
                
            if all_train_data:
                st.write("---")
                df_train = pd.DataFrame(all_train_data)
                csv_data = df_train.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Colab学習用CSV (v18.5_data.csv) をダウンロード",
                    data=csv_data, file_name="v18_5_data.csv", mime="text/csv", type="primary", use_container_width=True
                )

