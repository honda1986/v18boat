# -*- coding: utf-8 -*-
"""
v20.11 全艇スコア解析アプリ（インデント崩壊・完全防止フラット版）
"""
import re
import json
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=+9), 'JST')
ua_str = "Mozilla/5.0 (Windows NT 10.0)"
req_session = requests.Session()
req_session.headers.update({"User-Agent": ua_str})
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=2)
req_session.mount('https://', adapter)
req_session.mount('http://', adapter)

JCD_NAME = {1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",7:"蒲郡",8:"常滑",9:"津",10:"三国",11:"びわこ",12:"住之江",13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",17:"宮島",18:"徳山",19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"}

@dataclass
class Racer:
    name: str
    age: int
    cls_val: int
    weight: int
    f_count: int
    avg_st: float
    n_win: float
    n_2ren: float
    l_win: float
    l_2ren: float
    m_2ren: float
    b_2ren: float

def calc_extended_stats(racers: List[Racer]):
    a_w = sum(r.n_win for r in racers) / 6.0
    a_m = sum(r.m_2ren for r in racers) / 6.0
    a_s = sum(r.avg_st for r in racers) / 6.0
    w_r = sorted([r.n_win for r in racers], reverse=True)
    m_r = sorted([r.m_2ren for r in racers], reverse=True)
    s_r = sorted([r.avg_st for r in racers])
    stats = []
    for i, r in enumerate(racers):
        s_in = round(r.avg_st - racers[i-1].avg_st, 3) if i > 0 else 0.0
        w_in = round(r.n_win - racers[i-1].n_win, 2) if i > 0 else 0.0
        s_out = round(r.avg_st - racers[i+1].avg_st, 3) if i < 5 else 0.0
        w_out = round(r.n_win - racers[i+1].n_win, 2) if i < 5 else 0.0
        d = {"win_dev": round(r.n_win - a_w, 2), "motor_dev": round(r.m_2ren - a_m, 4), "st_dev": round(a_s - r.avg_st, 3), "win_rank": w_r.index(r.n_win) + 1, "motor_rank": m_r.index(r.m_2ren) + 1, "st_rank": s_r.index(r.avg_st) + 1, "st_diff_in": s_in, "win_diff_in": w_in, "st_diff_out": s_out, "win_diff_out": w_out}
        stats.append(d)
    return stats

@st.cache_resource
def load_lgb_model(filename: str):
    try:
        return lgb.Booster(model_file=filename)
    except:
        return None

lgb_model = load_lgb_model('lgb_score_v18_7.txt')
prob1_model = load_lgb_model('lgb_p1_v18_7.txt')
prob2_model = load_lgb_model('lgb_p2_v18_7.txt')
prob3_model = load_lgb_model('lgb_p3_v18_7.txt')

def get_lgb_features(r: Racer, lane: int, venue: str, rel: Dict):
    jcd_val = 1
    for k, v in JCD_NAME.items():
        if v == venue:
            jcd_val = k
            break
    f = [float(jcd_val), float(lane), float(r.cls_val), float(r.age), float(r.weight), float(r.f_count), float(r.avg_st), float(r.n_win), float(r.n_2ren), float(r.l_win), float(r.l_2ren), float(r.m_2ren), float(r.b_2ren), float(rel["win_dev"]), float(rel["motor_dev"]), float(rel["st_dev"]), float(rel["win_rank"]), float(rel["motor_rank"]), float(rel["st_rank"]), float(rel["st_diff_in"]), float(rel["win_diff_in"]), float(rel["st_diff_out"]), float(rel["win_diff_out"])]
    return f

def rank_all(racers: List[Racer], venue: str):
    out = []
    rel_stats = calc_extended_stats(racers)
    for i, r in enumerate(racers):
        lane = i + 1
        features = get_lgb_features(r, lane, venue, rel_stats[i])
        s = round(lgb_model.predict([features])[0] * 10, 2) if lgb_model else 0.0
        p1 = round(prob1_model.predict([features])[0] * 100, 1) if prob1_model else 0.0
        p2 = round(prob2_model.predict([features])[0] * 100, 1) if prob2_model else 0.0
        p3 = round(prob3_model.predict([features])[0] * 100, 1) if prob3_model else 0.0
        d = {"lane": lane, "racer": r, "score": s, "1着率": p1, "2着率": p2, "3着率": p3, "rel": rel_stats[i]}
        out.append(d)
    out.sort(key=lambda x: x["score"], reverse=True)
    lane1_p = None
    for x in out:
        if x["lane"] == 1:
            lane1_p = x["1着率"]
            break
    return out, lane1_p

def get_bet_probs(ranked: List[Dict]):
    p1 = {r["lane"]: max(0.1, r["1着率"]) for r in ranked}
    p2 = {r["lane"]: max(0.1, r["2着率"]) for r in ranked}
    p3 = {r["lane"]: max(0.1, r["3着率"]) for r in ranked}
    sum1 = sum(p1.values())
    for k in p1.keys():
        p1[k] = p1[k] / sum1
    bet_probs = []
    for l1 in range(1, 7):
        for l2 in range(1, 7):
            if l1 == l2: continue
            for l3 in range(1, 7):
                if l3 in (l1, l2): continue
                pr_1 = p1[l1]
                s2 = sum(p2[k] for k in range(1, 7) if k != l1)
                pr_2 = p2[l2] / s2 if s2 > 0 else 0
                s3 = sum(p3[k] for k in range(1, 7) if k not in (l1, l2))
                pr_3 = p3[l3] / s3 if s3 > 0 else 0
                prob = pr_1 * pr_2 * pr_3 * 100
                bet_probs.append({"bet": str(l1)+"-"+str(l2)+"-"+str(l3), "prob": round(prob, 2)})
    bet_probs.sort(key=lambda x: x["prob"], reverse=True)
    return bet_probs

def fetch_kyotei24_data(jcd: int, rno: int, dstr: str):
    url = "https://info.kyotei.fun/info-" + dstr + "-" + str(jcd).zfill(2) + "-" + str(rno) + ".html"
    try:
        r = req_session.get(url, timeout=7)
        r.encoding = r.apparent_encoding
        html = r.text if r.status_code == 200 else ""
    except:
        return None
    if not html or "出走表" not in html:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
    except:
        soup = BeautifulSoup(html, "html.parser")
    lane_to_rank = {}
    jyuni_divs = soup.find_all('div', class_='jyuni')
    has_res = False
    if len(jyuni_divs) >= 6:
        for i in range(6):
            txt = jyuni_divs[i].get_text(strip=True)
            if txt.isdigit():
                lane_to_rank[i+1] = txt
                has_res = True
    payoff = 0
    if has_res:
        p_div = soup.find('div', class_='race_result_end_label', string=re.compile('3連単'))
        if p_div and p_div.parent:
            m_s = p_div.parent.find('span', class_='race_result_end_money_num')
            if m_s:
                ptxt = m_s.get_text(strip=True).replace(',', '')
                if ptxt.isdigit():
                    payoff = int(ptxt)
    rd = [{"name": "選手"+str(i+1), "age": 30, "cls": 1, "weight": 50, "f": 0, "st": 0.17, "nw": 0.0, "n2": 0.0, "lw": 0.0, "l2": 0.0, "m2": 0.0, "b2": 0.0} for i in range(6)]
    cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    cur_lbl = ""
    for tr in soup.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if not tds: continue
        if len(tds) >= 7:
            cur_lbl = tds[0].get_text(strip=True).replace('\n', '').replace(' ', '')
            data_tds = tds[-6:]
        elif len(tds) == 6 and cur_lbl:
            data_tds = tds
        else:
            cur_lbl = ""
            continue
        for i in range(6):
            t_raw = data_tds[i].get_text(" ", strip=True)
            t_no = t_raw.replace(' ', '').replace('　', '').replace('\n', '')
            if "選手名" in cur_lbl:
                m_age = re.search(r'\((\d{2})\)', t_no)
                if m_age: rd[i]["age"] = int(m_age.group(1))
            elif "選手情報" in cur_lbl or "支部" in cur_lbl or "級" in cur_lbl:
                m_cls = re.search(r'([A12B]{2})', t_no)
                if m_cls: rd[i]["cls"] = cls_map.get(m_cls.group(1), 1)
                m_w = re.search(r'(\d+)kg', t_no, re.IGNORECASE)
                if m_w: rd[i]["weight"] = int(m_w.group(1))
            elif "全国" in cur_lbl and "勝率" in cur_lbl:
                m_2 = re.search(r'^([\d\.]+)', t_no)
                m_w = re.search(r'\(([\d\.]+)\)', t_no)
                if m_2:
                    v = float(m_2.group(1))
                    rd[i]["n2"] = v / 100.0 if v > 1.0 else v
                if m_w: rd[i]["nw"] = float(m_w.group(1))
            elif "当地" in cur_lbl and "勝率" in cur_lbl:
                m_2 = re.search(r'^([\d\.]+)', t_no)
                m_w = re.search(r'\(([\d\.]+)\)', t_no)
                if m_2:
                    v = float(m_2.group(1))
                    rd[i]["l2"] = v / 100.0 if v > 1.0 else v
                if m_w: rd[i]["lw"] = float(m_w.group(1))
            elif "モータ" in cur_lbl and "2連率" in cur_lbl:
                m = re.search(r'^([\d\.]+)', t_no)
                if m:
                    v = float(m.group(1))
                    rd[i]["m2"] = v / 100.0 if v > 1.0 else v
            elif "ボート" in cur_lbl and "2連率" in cur_lbl:
                m = re.search(r'^([\d\.]+)', t_no)
                if m:
                    v = float(m.group(1))
                    rd[i]["b2"] = v / 100.0 if v > 1.0 else v
            elif "平均ST" in cur_lbl:
                try: rd[i]["st"] = float(t_no)
                except: pass
            elif "フライング" in cur_lbl:
                try: rd[i]["f"] = int(t_no)
                except: pass
    if sum(x["nw"] for x in rd) == 0:
        return None
    racers = [Racer(name=x["name"], age=x["age"], cls_val=x["cls"], weight=x["weight"], f_count=x["f"], avg_st=x["st"], n_win=x["nw"], n_2ren=x["n2"], l_win=x["lw"], l_2ren=x["l2"], m_2ren=x["m2"], b_2ren=x["b2"]) for x in rd]
    return racers, lane_to_rank, payoff, has_res

def fetch_realtime_odds(jcd: int, rno: int, dstr: str):
    odds_dict = {}
    j_str = str(jcd).zfill(2)
    db_log = []
    t_url = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno=" + str(rno) + "&jcd=" + j_str + "&hd=" + dstr
    u_url = "http://uchisankaku.sakura.ne.jp/odds?jcd=" + j_str + "&rno=" + str(rno)
    k_url = "https://kyotei24.jp/odds/" + dstr + "/" + j_str + "/" + str(rno) + ".html"
    px = [("Uchi", u_url), ("K24", k_url), ("P1_Uchi", "https://api.allorigins.win/get?url=" + u_url), ("P2_Uchi", "https://api.codetabs.com/v1/proxy?quest=" + u_url), ("P1_Offi", "https://api.allorigins.win/get?url=" + t_url)]
    for name, url in px:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                db_log.append(name + ": E" + str(r.status_code))
                continue
            text = r.text
            if "allorigins" in url:
                try: text = r.json().get("contents", "")
                except: pass
            matches = re.findall(r'([1-6]-[1-6]-[1-6])[^\d]{1,20}?([1-9]\d{0,3}\.\d)', text)
            if matches:
                for bet, o_str in matches:
                    odds_dict[bet] = float(o_str)
                if len(odds_dict) > 10: return odds_dict, name + " OK"
            soup = BeautifulSoup(text, 'html.parser')
            odds_els = soup.find_all('td', class_='oddsPoint')
            if len(odds_els) >= 120:
                idx = 0
                for l1 in range(1, 7):
                    for l2 in range(1, 7):
                        if l1 == l2: continue
                        for l3 in range(1, 7):
                            if l3 in (l1, l2): continue
                            if idx < len(odds_els):
                                val = odds_els[idx].get_text(strip=True)
                                try: odds_dict[str(l1)+"-"+str(l2)+"-"+str(l3)] = float(val)
                                except: pass
                            idx += 1
                if len(odds_dict) > 10: return odds_dict, name + " OK"
            db_log.append(name + ": No Data")
        except Exception as e:
            db_log.append(name + ": Err")
    return {}, " | ".join(db_log)

def get_auto_bet_queue(bets_list, venue, r_idx, amount):
    queue = []
    for bet in bets_list:
        try:
            ptn = [int(x) for x in bet.split("-")]
            queue.append({"venue": venue, "race": str(r_idx), "pattern": ptn, "amount": str(amount)})
        except:
            pass
    return queue

def get_auto_bet_queue_from_df(df_bt, amount):
    queue = []
    for index, row in df_bt.iterrows():
        if row["点数"] > 0 and row["買い目"] != "見":
            bets = row["買い目"].split(",")
            for bet in bets:
                try:
                    ptn = [int(x) for x in bet.split("-")]
                    queue.append({"venue": row["場"], "race": str(row["R"]), "pattern": ptn, "amount": str(amount)})
                except:
                    pass
    return queue

st.set_page_config(page_title="v20.11 期待値ハンターAI", layout="wide")
st.sidebar.markdown("### ⚙️ 【直前用】期待値＆資金設定")
ev_threshold = st.sidebar.slider("回収率が何%以上を狙うか", 80, 200, 110, 5)
st.sidebar.markdown("### ⚙️ 【共通】フィルター設定")
prob_threshold = st.sidebar.slider("足切り確率(%)以上", 0.5, 20.0, 3.0, 0.5)
max_bets = st.sidebar.slider("最大購入点数(点)", 1, 12, 4, 1)
bet_amount = st.sidebar.number_input("1点の購入金額(円)", min_value=100, step=100, value=100)
st.title("🚤 v20.11 期待値ハンター (プロキシ・フラット版)")
st.caption("完全データ主導型AI / 📈 IPブロック突破・自動オッズスナイパー")
if not lgb_model:
    st.warning("⚠️ v18.7用のAIモデルが見つかりません。")

tab1, tab2 = st.tabs(["🚀 直前解析", "📊 バックテスト"])

with tab1:
    st.markdown("##### 🚨 オッズを取得し、期待値の高い買い目を抽出。")
    st.info("💡 注意: 「本日開催中で、まだ発走していないレース」を指定。")
    col1, col2 = st.columns(2)
    with col1: d_input = st.date_input("日付", value=datetime.now(JST).date())
    with col2: v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
    r_idx = st.selectbox("レース", options=list(range(1, 13)))
    if st.button("🔍 直前オッズ取得 ＆ 期待値解析", type="primary"):
        dstr = d_input.strftime("%Y%m%d")
        res = fetch_kyotei24_data(v_idx, r_idx, dstr)
        if res:
            racers, _, _, _ = res
            ranked, _ = rank_all(racers, JCD_NAME[v_idx])
            bet_probs = get_bet_probs(ranked)
            with st.spinner("海外中継サーバーを経由してIPブロックを突破中..."):
                realtime_odds, debug_msg = fetch_realtime_odds(v_idx, r_idx, dstr)
            if not realtime_odds:
                st.error("❌ オッズの取得に失敗しました。")
                st.warning("【原因調査】\n" + debug_msg)
            else:
                st.success("✅ 突破成功！解析 ＆ オッズ取得 完了！ (経由: " + debug_msg + ")")
                ev_results = []
                for bp in bet_probs:
                    bet = bp["bet"]
                    prob = bp["prob"]
                    odds = realtime_odds.get(bet, 0.0)
                    ev = (prob / 100.0) * odds * 100 
                    if prob >= prob_threshold: 
                        ev_results.append({"bet": bet, "prob": prob, "odds": odds, "ev": round(ev, 1)})
                ev_results.sort(key=lambda x: x["ev"], reverse=True)
                buy_bets_data = [item for item in ev_results if item["ev"] >= ev_threshold][:max_bets]
                buy_bets_list = [item["bet"] for item in buy_bets_data]
                auto_bet_queue = get_auto_bet_queue(buy_bets_list, JCD_NAME[v_idx], r_idx, bet_amount)
                st.subheader("🤖 全自動購入用データ")
                if auto_bet_queue:
                    st.code(json.dumps(auto_bet_queue, ensure_ascii=False, indent=4), language="json")
                else:
                    st.warning(f"※ 期待値 {ev_threshold}% を超える目はありません。")
                st.markdown("---")
                st.subheader("🎯 買い目別 期待値ランキング")
                disp_ev = []
                for item in ev_results[:15]: 
                    is_buy = "✅ 購入" if item["bet"] in buy_bets_list else "見送り"
                    disp_ev.append({"買い目": item["bet"], "AI確率": str(item['prob']) + "%", "直前オッズ": str(item['odds']) + "倍", "期待値(EV)": str(item['ev']) + "%", "判定": is_buy})
                st.dataframe(pd.DataFrame(disp_ev), use_container_width=True)
        else:
            st.error("出走表が取得できませんでした。")

with tab2:
    st.markdown("##### 📝 前日予想による過去の検証（確率のみの判定）")
    c1, c2, c3 = st.columns(3)
    with c1: bt_start = st.date_input("開始日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with c2: bt_end = st.date_input("終了日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with c3: bt_v_idx = st.selectbox("場を指定", options=[0] + list(JCD_NAME.keys()), format_func=lambda x: "全国（すべて）" if x==0 else JCD_NAME[x])
    if st.button("📊 バックテスト実行 ＆ JSON生成", type="primary"):
        days = [(bt_start + timedelta(days=i)).strftime("%Y%m%d") for i in range((bt_end - bt_start).days + 1)]
        tasks = [(d, j, r) for d in days for j in ([bt_v_idx] if bt_v_idx != 0 else list(range(1, 25))) for r in range(1, 13)]
        matches = []
        prog = st.progress(0.0)
        def analyze_race(d, j, r):
            res = fetch_kyotei24_data(j, r, d)
            if not res: return None
            racers, lane_to_rank, payoff, has_result = res
            venue_name = JCD_NAME.get(j, "不明")
            ranked, _ = rank_all(racers, venue_name)
            bet_probs = get_bet_probs(ranked)
            buy_bets = [bp["bet"] for bp in bet_probs if bp["prob"] >= prob_threshold][:max_bets]
            if not has_result:
                h_str, p_disp, a_res, h_amt = "⏳", "-", "結果待ち" if buy_bets else "見送り", 0
            else:
                a_res = ""
                r1 = next((k for k, v in lane_to_rank.items() if str(v) == '1'), None)
                r2 = next((k for k, v in lane_to_rank.items() if str(v) == '2'), None)
                r3 = next((k for k, v in lane_to_rank.items() if str(v) == '3'), None)
                if r1 and r2 and r3: a_res = str(r1) + "-" + str(r2) + "-" + str(r3)
                if not buy_bets:
                    h_str, p_disp, h_amt = "ー", "(" + str(payoff) + ")", 0
                else:
                    hit = a_res in buy_bets
                    h_str, p_disp, h_amt = "🎯" if hit else "❌", payoff, payoff if hit else 0
            return {"日付": d, "場": venue_name, "R": r, "買い目": ",".join(buy_bets) if buy_bets else "見", "点数": len(buy_bets), "結果": a_res, "的中": h_str, "払戻金": p_disp, "_hit_amount": h_amt}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
            fut_dict = {exe.submit(analyze_race, d, j, r): (d, j, r) for d, j, r in tasks}
            done_cnt = 0
            for fut in concurrent.futures.as_completed(fut_dict):
                done_cnt += 1
                if len(tasks) > 0: prog.progress(done_cnt / len(tasks))
                res = fut.result()
                if res: matches.append(res)
        matches.sort(key=lambda x: (x["日付"], x["場"], x["R"]))
        if matches:
            df_bt = pd.DataFrame(matches)
            bet_races = [m for m in matches if m["点数"] > 0 and m["的中"] in ["🎯", "❌", "⏳"]]
            if bet_races:
                total_inv = sum(m["点数"] * bet_amount for m in bet_races if m["的中"] != "⏳")
                total_ret = sum(m["_hit_amount"] * (bet_amount / 100) for m in bet_races if m["的中"] != "⏳")
                st.success(f"🔥 勝負対象: {len(bet_races)}件 (見送り: {len([m for m in matches if m['点数']==0])}件)")
                if total_inv > 0:
                    st.info(f"💰 **総投資**: {total_inv:,.0f}円 / **総回収**: {total_ret:,.0f}円 (回収率: {(total_ret / total_inv) * 100:.1f}%)")
                st.dataframe(df_bt[["日付", "場", "R", "買い目", "結果", "的中", "払戻金"]], use_container_width=True)
                auto_bet_queue = get_auto_bet_queue_from_df(df_bt, bet_amount)
                if auto_bet_queue:
                    st.markdown("---")
                    st.subheader("🤖 全自動購入用データ")
                    st.code(json.dumps(auto_bet_queue, ensure_ascii=False, indent=4), language="json")
