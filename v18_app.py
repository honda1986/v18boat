# -*- coding: utf-8 -*-
"""
v20.10 全艇スコア解析アプリ（中継プロキシ突破・完全破壊耐性版）
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

adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=2
)
req_session.mount('https://', adapter)
req_session.mount('http://', adapter)

JCD_NAME = {}
JCD_NAME[1] = "桐生"
JCD_NAME[2] = "戸田"
JCD_NAME[3] = "江戸川"
JCD_NAME[4] = "平和島"
JCD_NAME[5] = "多摩川"
JCD_NAME[6] = "浜名湖"
JCD_NAME[7] = "蒲郡"
JCD_NAME[8] = "常滑"
JCD_NAME[9] = "津"
JCD_NAME[10] = "三国"
JCD_NAME[11] = "びわこ"
JCD_NAME[12] = "住之江"
JCD_NAME[13] = "尼崎"
JCD_NAME[14] = "鳴門"
JCD_NAME[15] = "丸亀"
JCD_NAME[16] = "児島"
JCD_NAME[17] = "宮島"
JCD_NAME[18] = "徳山"
JCD_NAME[19] = "下関"
JCD_NAME[20] = "若松"
JCD_NAME[21] = "芦屋"
JCD_NAME[22] = "福岡"
JCD_NAME[23] = "唐津"
JCD_NAME[24] = "大村"

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
        s_in = 0.0
        w_in = 0.0
        s_out = 0.0
        w_out = 0.0
        
        if i > 0:
            s_in = round(r.avg_st - racers[i-1].avg_st, 3)
            w_in = round(r.n_win - racers[i-1].n_win, 2)
        if i < 5:
            s_out = round(r.avg_st - racers[i+1].avg_st, 3)
            w_out = round(r.n_win - racers[i+1].n_win, 2)
            
        d = {}
        d["win_dev"] = round(r.n_win - a_w, 2)
        d["motor_dev"] = round(r.m_2ren - a_m, 4)
        d["st_dev"] = round(a_s - r.avg_st, 3)
        d["win_rank"] = w_r.index(r.n_win) + 1
        d["motor_rank"] = m_r.index(r.m_2ren) + 1
        d["st_rank"] = s_r.index(r.avg_st) + 1
        d["st_diff_in"] = s_in
        d["win_diff_in"] = w_in
        d["st_diff_out"] = s_out
        d["win_diff_out"] = w_out
        stats.append(d)
    return stats

@st.cache_resource
def load_lgb_model(filename: str):
    try:
        return lgb.Booster(model_file=filename)
    except:
        return None

lgb_model   = load_lgb_model('lgb_score_v18_7.txt')
prob1_model = load_lgb_model('lgb_p1_v18_7.txt')
prob2_model = load_lgb_model('lgb_p2_v18_7.txt')
prob3_model = load_lgb_model('lgb_p3_v18_7.txt')

def get_lgb_features(r: Racer, lane: int, venue: str, rel: Dict):
    jcd_val = 1
    for k, v in JCD_NAME.items():
        if v == venue:
            jcd_val = k
            break
            
    f = []
    f.append(float(jcd_val))
    f.append(float(lane))
    f.append(float(r.cls_val))
    f.append(float(r.age))
    f.append(float(r.weight))
    f.append(float(r.f_count))
    f.append(float(r.avg_st))
    f.append(float(r.n_win))
    f.append(float(r.n_2ren))
    f.append(float(r.l_win))
    f.append(float(r.l_2ren))
    f.append(float(r.m_2ren))
    f.append(float(r.b_2ren))
    f.append(float(rel["win_dev"]))
    f.append(float(rel["motor_dev"]))
    f.append(float(rel["st_dev"]))
    f.append(float(rel["win_rank"]))
    f.append(float(rel["motor_rank"]))
    f.append(float(rel["st_rank"]))
    f.append(float(rel["st_diff_in"]))
    f.append(float(rel["win_diff_in"]))
    f.append(float(rel["st_diff_out"]))
    f.append(float(rel["win_diff_out"]))
    return f

def rank_all(racers: List[Racer], venue: str):
    out = []
    rel_stats = calc_extended_stats(racers)
    for i, r in enumerate(racers):
        lane = i + 1
        features = get_lgb_features(r, lane, venue, rel_stats[i])
        s = 0.0
        p1 = 0.0
        p2 = 0.0
        p3 = 0.0
        
        if lgb_model:
            s = round(lgb_model.predict([features])[0] * 10, 2)
        if prob1_model:
            p1 = round(prob1_model.predict([features])[0] * 100, 1)
        if prob2_model:
            p2 = round(prob2_model.predict([features])[0] * 100, 1)
        if prob3_model:
            p3 = round(prob3_model.predict([features])[0] * 100, 1)
            
        d = {}
        d["lane"] = lane
        d["racer"] = r
        d["score"] = s
        d["1着率"] = p1
        d["2着率"] = p2
        d["3着率"] = p3
        d["rel"] = rel_stats[i]
        out.append(d)
        
    out.sort(key=lambda x: x["score"], reverse=True)
    lane1_p = None
    for x in out:
        if x["lane"] == 1:
            lane1_p = x["1着率"]
            break
    return out, lane1_p

def get_bet_probs(ranked: List[Dict]):
    p1 = {}
    p2 = {}
    p3 = {}
    for r in ranked:
        p1[r["lane"]] = max(0.1, r["1着率"])
        p2[r["lane"]] = max(0.1, r["2着率"])
        p3[r["lane"]] = max(0.1, r["3着率"])
        
    sum1 = sum(p1.values())
    for k in p1.keys():
        p1[k] = p1[k] / sum1
        
    bet_probs = []
    
    for l1 in range(1, 7):
        for l2 in range(1, 7):
            if l1 == l2:
                continue
            for l3 in range(1, 7):
                if l3 in (l1, l2):
                    continue
                pr_1 = p1[l1]
                
                s2 = 0.0
                for k in range(1, 7):
                    if k != l1: s2 += p2[k]
                pr_2 = p2[l2] / s2 if s2 > 0 else 0
                
                s3 = 0.0
                for k in range(1, 7):
                    if k not in (l1, l2): s3 += p3[k]
                pr_3 = p3[l3] / s3 if s3 > 0 else 0
                
                prob = pr_1 * pr_2 * pr_3 * 100
                b_s = str(l1) + "-" + str(l2) + "-" + str(l3)
                
                d = {}
                d["bet"] = b_s
                d["prob"] = round(prob, 2)
                bet_probs.append(d)
                
    bet_probs.sort(key=lambda x: x["prob"], reverse=True)
    return bet_probs

def fetch_kyotei24_data(jcd: int, rno: int, dstr: str):
    u = "https://info.kyotei.fun/info-" + dstr
    url = u + "-" + str(jcd).zfill(2) + "-" + str(rno) + ".html"
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
        ptn = re.compile('3連単')
        p_div = soup.find('div', class_='race_result_end_label', string=ptn)
        if p_div and p_div.parent:
            m_s = p_div.parent.find('span', class_='race_result_end_money_num')
            if m_s:
                ptxt = m_s.get_text(strip=True).replace(',', '')
                if ptxt.isdigit():
                    payoff = int(ptxt)
                    
    rd = []
    for i in range(6):
        d = {}
        d["name"] = "選手" + str(i+1)
        d["age"] = 30
        d["cls"] = 1
        d["weight"] = 50
        d["f"] = 0
        d["st"] = 0.17
        d["nw"] = 0.0
        d["n2"] = 0.0
        d["lw"] = 0.0
        d["l2"] = 0.0
        d["m2"] = 0.0
        d["b2"] = 0.0
        rd.append(d)
        
    cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    cur_lbl = ""
    
    for tr in soup.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if not tds:
            continue
            
        if len(tds) >= 7:
            txt_0 = tds[0].get_text(strip=True)
            cur_lbl = txt_0.replace('\n', '').replace(' ', '')
            data_tds = tds[-6:]
        elif len(tds) == 6 and cur_lbl:
            data_tds = tds
        else:
            cur_lbl = ""
            continue
            
        for i in range(6):
            t_raw = data_tds[i].get_text(" ", strip=True)
            t_no = t_raw.replace(' ', '').replace('　', '')
            t_no = t_no.replace('\n', '')
            
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
                    
    nws = sum(x["nw"] for x in rd)
    if nws == 0:
        return None
        
    racers = []
    for x in rd:
        obj = Racer(
            name=x["name"], age=x["age"], cls_val=x["cls"],
            weight=x["weight"], f_count=x["f"], avg_st=x["st"],
            n_win=x["nw"], n_2ren=x["n2"], l_win=x["lw"],
            l_2ren=x["l2"], m_2ren=x["m2"], b_2ren=x["b2"]
        )
        racers.append(obj)
        
    return racers, lane_to_rank, payoff, has_res

def fetch_realtime_odds(jcd: int, rno: int, dstr: str):
    odds_dict = {}
    j_str = str(jcd).zfill(2)
    db_log = []
    
    t_url = "https://www.boatrace.jp/owpc/pc/race/odds3t"
    t_url += "?rno=" + str(rno) + "&jcd=" + j_str + "&hd=" + dstr
    
    u_url = "http://uchisankaku.sakura.ne.jp/odds"
    u_url += "?jcd=" + j_str + "&rno=" + str(rno)
    
    px = []
    px.append(("Direct_Uchi", u_url))
    px.append(("Proxy1_Uchi", "https://api.allorigins.win/get?url=" + u_url))
    px.append(("Proxy2_Uchi", "https://api.codetabs.com/v1/proxy?quest=" + u_url))
    px.append(("Proxy1_Offi", "https://api.allorigins.win/get?url=" + t_url))
    px.append(("Proxy2_Offi", "https://api.codetabs.com/v1/proxy?quest=" + t_url))

    for name, url in px:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                db_log.append(name + ": E" + str(r.status_code))
                continue
                
            text = r.text
            if "allorigins" in url:
                try:
                    data = r.json()
                    text = data.get("contents", "")
                except:
                    pass
                    
            ptn = r'([1-6]-[1-6]-[1-6])[^\d]{1,20}?([1-9]\d{0,3}\.\d)'
            matches = re.findall(ptn, text)
            if matches:
                for bet, o_str in matches:
                    odds_dict[bet] = float(o_str)
                if len(odds_dict) > 10:
                    return odds_dict, name + " OK"
            
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
                                try:
                                    b_s = str(l1)+"-"+str(l2)+"-"+str(l3)
                                    odds_dict[b_s] = float(val)
                                except: pass
                            idx += 1
                if len(odds_dict) > 10:
                    return odds_dict, name + " OK"
                    
            db_log.append(name + ": No Data")
        except Exception as e:
            db_log.append(name + ": Err")

    return {}, " | ".join(db_log)

st.set_page_config(page_title="v20.10 期待値ハンターAI", layout="wide")

st.sidebar.markdown("### ⚙️ 【直前用】期待値＆資金設定")
ev_threshold = st.sidebar.slider("回収率が何%以上を狙うか", 80, 200, 110, 5)

st.sidebar.markdown("### ⚙️ 【共通】フィルター設定")
prob_threshold = st.sidebar.slider("足切り確率(%)以上", 0.5, 20.0, 3.0, 0.5)
max_bets = st.sidebar.slider("最大購入点数(点)", 1, 12, 4, 1)
bet_amount = st.sidebar.number_input("1点の購入金額(円)", min_value=100, step=100, value=100)

st.title("🚤 v20.10 期待値ハンター (プロキシ突破版)")
st.caption("完全データ主導型AI / 📈 IPブロック突破・自動オッズスナイパー")

if not lgb_model:
    st.warning("⚠️ v18.7用のAIモデルが見つかりません。")

tab1, tab2 = st.tabs(["🚀 直前解析", "📊 バックテスト"])

with tab1:
    st.markdown("##### 🚨 オッズを取得し、期待値の高い買い目を抽出。")
    st.info("💡 注意: 「本日開催中で、まだ発走していないレース」を指定。")
    
    col1, col2 = st.columns(2)
    with col1:
        d_input = st.date_input("日付", value=datetime.now(JST).date())
    with col2:
        v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
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
                st.info("※中継サーバーも弾かれているか、レースが終了している可能性があります。")
            else:
                st.success("✅ 突破成功！解析 ＆ オッズ取得 完了！ (経由: " + debug_msg + ")")
                
                ev_results = []
                for bp in bet_probs:
                    bet = bp["bet"]
                    prob = bp["prob"]
                    odds = realtime_odds.get(bet, 0.0)
                    ev = (prob / 100.0) * odds * 100 
                    
                    if prob >= prob_threshold: 
                        d = {}
                        d["bet"] = bet
                        d["prob"] = prob
                        d["odds"] = odds
                        d["ev"] = round(ev, 1)
                        ev_results.append(d)
                
                ev_results.sort(key=lambda x: x["ev"], reverse=True)
                
                buy_bets_data = []
                for item in ev_results:
                    if item["ev"] >= ev_threshold:
                        buy_bets_data.append(item)
                buy_bets_data = buy_bets_data[:max_bets]
                
                buy_bets_list = [item["bet"] for item in buy_bets_data]

                auto_bet_queue = []
                for bet in buy_bets_list:
                    try:
                        pattern = [int(x) for x in bet.split("-")]
                        qd = {}
                        qd["venue"] = JCD_NAME[v_idx]
                        qd["race"] = str(r_idx)
                        qd["pattern"] = pattern
                        qd["amount"] = str(bet_amount)
                        auto_bet_queue.append(qd)
                    except:
                        pass

                st.subheader("🤖 全自動購入用データ")
                if auto_bet_queue:
                    st.code(json.dumps(auto_bet_queue, ensure_ascii=False, indent=4), language="json")
                else:
                    m_n = f"※ 期待値 {ev_threshold}% を超える目はありません。"
                    st.warning(m_n)
                
                st.markdown("---")
                st.subheader("🎯 買い目別 期待値ランキング")
                disp_ev = []
                for item in ev_results[:15]: 
                    is_buy = "✅ 購入" if item["bet"] in buy_bets_list else "見送り"
                    d = {}
                    d["買い目"] = item["bet"]
                    d["AI確率"] = str(item['prob']) + "%"
                    d["直前オッズ"] = str(item['odds']) + "倍"
                    d["期待値(EV)"] = str(item['ev']) + "%"
                    d["判定"] = is_buy
                    disp_ev.append(d)
                st.dataframe(pd.DataFrame(disp_ev), use_container_width=True)
        else:
            st.error("出走表が取得できませんでした。")

with tab2:
    st.markdown("##### 📝 前日予想による過去の検証（確率のみの判定）")
    c1, c2, c3 = st.columns(3)
    with c1:
        bt_start = st.date_input("開始日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with c2:
        bt_end = st.date_input("終了日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with c3:
        bt_v_idx = st.selectbox("場を指定", options=[0] + list(JCD_NAME.keys()), format_func=lambda x: "全国（すべて）" if x==0 else JCD_NAME[x])

    if st.button("📊 バックテスト実行 ＆ JSON生成", type="primary"):
        days = []
        for i in range((bt_end - bt_start).days + 1):
            dt = bt_start + timedelta(days=i)
            days.append(dt.strftime("%Y%m%d"))
            
        tasks = []
        for dstr in days:
            target_j = [bt_v_idx] if bt_v_idx != 0 else list(range(1, 25))
            for j in target_j:
                for r in range(1, 13):
                    tasks.append((dstr, j, r))

        matches = []
        prog = st.progress(0.0)
        
        def analyze_race(d, j, r):
            res = fetch_kyotei24_data(j, r, d)
            if not res: return None
            
            racers, lane_to_rank, payoff, has_result = res
            venue_name = JCD_NAME.get(j, "不明")
            ranked, _ = rank_all(racers, venue_name)
            
            bet_probs = get_bet_probs(ranked)
            filtered_bets = []
            for bp in bet_probs:
                if bp["prob"] >= prob_threshold:
                    filtered_bets.append(bp["bet"])
                    
            buy_bets = filtered_bets[:max_bets]
            
            if not has_result:
                h_str = "⏳"
                p_disp = "-"
                a_res = "結果待ち" if buy_bets else "見送り"
                h_amt = 0
            else:
                a_res = ""
                r1 = None
                r2 = None
                r3 = None
                for k, v in lane_to_rank.items():
          
