# -*- coding: utf-8 -*-
"""
v20.0 全艇スコア解析アプリ（期待値EVベース判定版）

【v19.3からの主な変更点】
  1. 公式サイト(boatrace.jp)から3連単オッズを取得する fetch_odds3t を追加
  2. 買い目選定を「確率順」から「期待値(EV)順」に刷新 … 回収率改善の本丸
     EV = 予想確率 × オッズ。EV>1.0で理論プラス、それ未満は買うほど損。
  3. オッズ表のセル並び順を、確定済レースの払戻と突き合わせて自動判定する
     verify_odds_ordering を追加（初回に1回だけ実行を推奨）
  4. 直前情報(展示タイム・風)を参考表示（※現行モデルは23特徴量固定のため未使用）

【既存モデルは無改変】get_lgb_features は23特徴量のまま。モデルファイルもそのまま。
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
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
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
JCD_FROM_NAME = {v: k for k, v in JCD_NAME.items()}

@dataclass
class Racer:
    name: str; age: int; cls_val: int; weight: int; f_count: int; avg_st: float
    n_win: float; n_2ren: float; l_win: float; l_2ren: float; m_2ren: float; b_2ren: float

def calc_extended_stats(racers: List[Racer]) -> List[Dict]:
    avg_win = sum(r.n_win for r in racers) / 6.0
    avg_motor = sum(r.m_2ren for r in racers) / 6.0
    avg_st = sum(r.avg_st for r in racers) / 6.0
    win_rates = sorted([r.n_win for r in racers], reverse=True)
    motors = sorted([r.m_2ren for r in racers], reverse=True)
    sts = sorted([r.avg_st for r in racers])

    stats = []
    for i, r in enumerate(racers):
        st_diff_in = round(r.avg_st - racers[i-1].avg_st, 3) if i > 0 else 0.0
        win_diff_in = round(r.n_win - racers[i-1].n_win, 2) if i > 0 else 0.0
        st_diff_out = round(r.avg_st - racers[i+1].avg_st, 3) if i < 5 else 0.0
        win_diff_out = round(r.n_win - racers[i+1].n_win, 2) if i < 5 else 0.0
        stats.append({
            "win_dev": round(r.n_win - avg_win, 2), "motor_dev": round(r.m_2ren - avg_motor, 4), "st_dev": round(avg_st - r.avg_st, 3),
            "win_rank": win_rates.index(r.n_win) + 1, "motor_rank": motors.index(r.m_2ren) + 1, "st_rank": sts.index(r.avg_st) + 1,
            "st_diff_in": st_diff_in, "win_diff_in": win_diff_in, "st_diff_out": st_diff_out, "win_diff_out": win_diff_out
        })
    return stats

@st.cache_resource
def load_lgb_model(filename: str):
    try: return lgb.Booster(model_file=filename)
    except: return None

lgb_model   = load_lgb_model('lgb_score_v18_7.txt')
prob1_model = load_lgb_model('lgb_p1_v18_7.txt')
prob2_model = load_lgb_model('lgb_p2_v18_7.txt')
prob3_model = load_lgb_model('lgb_p3_v18_7.txt')

def get_lgb_features(r: Racer, lane: int, venue: str, rel: Dict) -> list:
    # ※ 23特徴量固定。並び・数を変えると既存モデルが壊れるため変更しないこと。
    jcd = JCD_FROM_NAME.get(venue, 1)
    return [
        float(jcd), float(lane), float(r.cls_val), float(r.age), float(r.weight), float(r.f_count), float(r.avg_st),
        float(r.n_win), float(r.n_2ren), float(r.l_win), float(r.l_2ren), float(r.m_2ren), float(r.b_2ren),
        float(rel["win_dev"]), float(rel["motor_dev"]), float(rel["st_dev"]), float(rel["win_rank"]), float(rel["motor_rank"]), float(rel["st_rank"]),
        float(rel["st_diff_in"]), float(rel["win_diff_in"]), float(rel["st_diff_out"]), float(rel["win_diff_out"])
    ]

def rank_all(racers: List[Racer], venue: str) -> Tuple[List[Dict], Optional[float]]:
    out = []
    rel_stats = calc_extended_stats(racers)
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

def get_bet_probs(ranked: List[Dict]) -> List[Dict]:
    p1 = {r["lane"]: max(0.1, r["1着率"]) for r in ranked}
    p2 = {r["lane"]: max(0.1, r["2着率"]) for r in ranked}
    p3 = {r["lane"]: max(0.1, r["3着率"]) for r in ranked}

    sum1 = sum(p1.values())
    p1 = {k: v / sum1 for k, v in p1.items()}

    bet_probs = []
    for l1 in range(1, 7):
        for l2 in range(1, 7):
            if l1 == l2: continue
            for l3 in range(1, 7):
                if l3 in (l1, l2): continue

                prob_1st = p1[l1]
                sum2 = sum(p2[k] for k in range(1, 7) if k != l1)
                prob_2nd = p2[l2] / sum2 if sum2 > 0 else 0
                sum3 = sum(p3[k] for k in range(1, 7) if k not in (l1, l2))
                prob_3rd = p3[l3] / sum3 if sum3 > 0 else 0

                prob = prob_1st * prob_2nd * prob_3rd * 100
                bet_probs.append({"bet": f"{l1}-{l2}-{l3}", "prob": round(prob, 2)})

    bet_probs.sort(key=lambda x: x["prob"], reverse=True)
    return bet_probs

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
            if txt.isdigit(): lane_to_rank[i+1] = txt; has_result = True
    payoff = 0
    if has_result:
        payoff_div = soup.find('div', class_='race_result_end_label', string=re.compile('3連単'))
        if payoff_div and payoff_div.parent:
            money_span = payoff_div.parent.find('span', class_='race_result_end_money_num')
            if money_span:
                ptxt = money_span.get_text(strip=True).replace(',', '')
                if ptxt.isdigit(): payoff = int(ptxt)
    rd = [{"name": f"選手{i+1}", "age": 30, "cls": 1, "weight": 50, "f": 0, "st": 0.17, "nw": 0.0, "n2": 0.0, "lw": 0.0, "l2": 0.0, "m2": 0.0, "b2": 0.0} for i in range(6)]
    cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    current_label = ""
    for tr in soup.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if not tds: continue
        if len(tds) >= 7: current_label = tds[0].get_text(strip=True).replace('\n', '').replace(' ', ''); data_tds = tds[-6:]
        elif len(tds) == 6 and current_label: data_tds = tds
        else: current_label = ""; continue
        for i in range(6):
            txt_raw = data_tds[i].get_text(" ", strip=True)
            txt_nospace = txt_raw.replace(' ', '').replace('　', '').replace('\n', '')
            if "選手名" in current_label:
                m_age = re.search(r'\((\d{2})\)', txt_nospace)
                if m_age: rd[i]["age"] = int(m_age.group(1))
            elif "選手情報" in current_label or "支部" in current_label or "級" in current_label:
                m_cls = re.search(r'([A12B]{2})', txt_nospace)
                if m_cls: rd[i]["cls"] = cls_map.get(m_cls.group(1), 1)
                m_w = re.search(r'(\d+)kg', txt_nospace, re.IGNORECASE)
                if m_w: rd[i]["weight"] = int(m_w.group(1))
            elif "全国" in current_label and "勝率" in current_label:
                m_2 = re.search(r'^([\d\.]+)', txt_nospace); m_w = re.search(r'\(([\d\.]+)\)', txt_nospace)
                if m_2: rd[i]["n2"] = float(m_2.group(1))/100.0 if float(m_2.group(1))>1.0 else float(m_2.group(1))
                if m_w: rd[i]["nw"] = float(m_w.group(1))
            elif "当地" in current_label and "勝率" in current_label:
                m_2 = re.search(r'^([\d\.]+)', txt_nospace); m_w = re.search(r'\(([\d\.]+)\)', txt_nospace)
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
    racers = [Racer(name=x["name"], age=x["age"], cls_val=x["cls"], weight=x["weight"], f_count=x["f"], avg_st=x["st"], n_win=x["nw"], n_2ren=x["n2"], l_win=x["lw"], l_2ren=x["l2"], m_2ren=x["m2"], b_2ren=x["b2"]) for x in rd]
    return racers, lane_to_rank, payoff, has_result

# ============================================================
# 🆕 公式オッズ取得 ＋ 期待値(EV)ロジック
# ============================================================

def _build_combo_orderings() -> Dict[str, List[str]]:
    """odds3t ページの td.oddsPoint(120個) の並び順 候補。
    公式HTMLのセル順は環境/改修で変わり得るため複数候補を用意し、
    verify_odds_ordering() で確定レースの払戻と突き合わせて正解を自動判定する。"""
    boats = [1, 2, 3, 4, 5, 6]
    orderings: Dict[str, List[str]] = {}

    # 候補A: 1着→2着→3着 の素直な辞書順（1着が外ループ）
    orderings["A"] = [f"{a}-{b}-{c}" for a in boats for b in boats for c in boats
                      if len({a, b, c}) == 3]

    # 候補B: 公式テーブル構造（1着=列を横に / 2-3着=縦）に対応。
    #         DOM行優先 = ブロック内縦位置(外) × 1着(内ループ)
    comboB: List[str] = []
    for p in range(20):                         # 各1着ブロック内の縦位置 0..19
        m, n = divmod(p, 4)
        for f in boats:                         # 1着が内側で回る（列方向）
            others = [x for x in boats if x != f]    # 昇順5艇
            second = others[m]
            rest = [x for x in others if x != second]  # 残り4艇
            third = rest[n]
            comboB.append(f"{f}-{second}-{third}")
    orderings["B"] = comboB

    return orderings

_COMBO_ORDERINGS = _build_combo_orderings()
DEFAULT_ORDERING = "A"

def _get_active_ordering() -> str:
    """メインスレッドからのみ呼ぶこと（ワーカースレッドからは引数で渡す）。"""
    try:
        return st.session_state.get("odds_ordering", DEFAULT_ORDERING)
    except Exception:
        return DEFAULT_ORDERING

def _fetch_odds_values(jcd: int, rno: int, dstr: str) -> List[float]:
    """odds3t ページの td.oddsPoint を出現順に取得（生データ120個想定）。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={dstr}"
    try:
        r = req_session.get(url, timeout=8)
        r.encoding = r.apparent_encoding
        if r.status_code != 200:
            return []
        try:
            soup = BeautifulSoup(r.text, "lxml")
        except Exception:
            soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return []
    vals: List[float] = []
    for cell in soup.select("td.oddsPoint"):
        txt = cell.get_text(strip=True).replace(",", "")
        try:
            vals.append(float(txt))
        except Exception:
            vals.append(0.0)    # 欠場「.」など数値化できないセル
    return vals

def fetch_odds3t(jcd: int, rno: int, dstr: str, ordering: Optional[str] = None) -> Dict[str, float]:
    """3連単オッズを {'1-2-3': 7.5, ...} で返す。120個揃わなければ {}（未確定/失敗）。"""
    vals = _fetch_odds_values(jcd, rno, dstr)
    if len(vals) != 120:
        return {}
    name = ordering or _get_active_ordering()
    combos = _COMBO_ORDERINGS.get(name, _COMBO_ORDERINGS["A"])
    return {c: o for c, o in zip(combos, vals)}

def verify_odds_ordering(sample_tasks: List[Tuple[str, int, int]]) -> Dict[str, Tuple[float, int]]:
    """確定済レースで「オッズ[勝ち目]×100 ≒ 払戻金」になる並び順を判定する。
    返り値: {並び順名: (平均相対誤差, 検証できたレース数)}。誤差が最小の候補が正解。"""
    scores: Dict[str, List[float]] = {name: [] for name in _COMBO_ORDERINGS}
    for (d, j, r) in sample_tasks:
        res = fetch_kyotei24_data(j, r, d)
        if not res:
            continue
        racers, lane_to_rank, payoff, has_result = res
        if not has_result or not payoff or payoff <= 0:
            continue
        r1 = next((k for k, v in lane_to_rank.items() if str(v) == '1'), None)
        r2 = next((k for k, v in lane_to_rank.items() if str(v) == '2'), None)
        r3 = next((k for k, v in lane_to_rank.items() if str(v) == '3'), None)
        if not (r1 and r2 and r3):
            continue
        winner = f"{r1}-{r2}-{r3}"
        vals = _fetch_odds_values(j, r, d)
        if len(vals) != 120:
            continue
        for name, combos in _COMBO_ORDERINGS.items():
            o = dict(zip(combos, vals)).get(winner, 0.0)
            if o > 0:
                scores[name].append(abs(o * 100 - payoff) / payoff)
    return {name: ((sum(e) / len(e)) if e else 9.99, len(e)) for name, e in scores.items()}

def select_bets(ranked: List[Dict], odds_map: Dict[str, float], mode: str,
                ev_th: float, min_prob: float, prob_th: float, max_n: int) -> Tuple[List[Dict], List[Dict]]:
    """買い目を選定する。返り値=(全候補[prob/odds/ev付与], 採用買い目リスト)。
    mode='ev'  : オッズ取得済 & 予想確率>=min_prob & EV>=ev_th を満たすものをEV降順に採用
    mode='prob': 予想確率>=prob_th を確率降順に採用（旧方式・オッズ不問）"""
    bet_probs = get_bet_probs(ranked)
    for bp in bet_probs:
        o = odds_map.get(bp["bet"], 0.0)
        bp["odds"] = o
        bp["ev"] = round(bp["prob"] / 100.0 * o, 3) if o > 0 else 0.0
    if mode == "ev":
        cand = [bp for bp in bet_probs
                if bp["odds"] > 0 and bp["prob"] >= min_prob and bp["ev"] >= ev_th]
        cand.sort(key=lambda x: x["ev"], reverse=True)
    else:
        cand = [bp for bp in bet_probs if bp["prob"] >= prob_th]
        cand.sort(key=lambda x: x["prob"], reverse=True)
    return bet_probs, cand[:max_n]

def fetch_beforeinfo(jcd: int, rno: int, dstr: str) -> Dict:
    """直前情報(展示タイム・風)を best-effort 取得。★表示専用★（モデル未使用）。
    ここを特徴量に使うには23→Nへ拡張して再学習が必要。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={dstr}"
    info: Dict = {"tenji": {}, "weather": {}}
    try:
        r = req_session.get(url, timeout=8)
        r.encoding = r.apparent_encoding
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        return {}
    # 展示タイム（おおむね 6.xx〜7.xx 秒）を各艇行から拾う簡易抽出
    try:
        for tr in soup.select("table tbody tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            head = tds[0].get_text(strip=True)
            if head.isdigit() and 1 <= int(head) <= 6:
                lane = int(head)
                for td in tds:
                    t = td.get_text(strip=True)
                    if re.fullmatch(r'[67]\.\d{2}', t):
                        info["tenji"][lane] = float(t)
                        break
    except Exception:
        pass
    # 天候・風・波
    try:
        wtxt = soup.get_text(" ", strip=True)
        mw = re.search(r'風速\s*([\d.]+)\s*m', wtxt)
        if mw: info["weather"]["風速(m)"] = float(mw.group(1))
        mt = re.search(r'気温\s*([\d.]+)', wtxt)
        if mt: info["weather"]["気温"] = float(mt.group(1))
        mv = re.search(r'波高\s*([\d.]+)\s*cm', wtxt)
        if mv: info["weather"]["波高(cm)"] = float(mv.group(1))
    except Exception:
        pass
    return info

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="v20.0 期待値EVベース判定版", layout="wide")

if not lgb_model:
    st.warning("⚠️ v18.7用のAIモデルが見つかりません。")

# --- サイドバー ---
st.sidebar.markdown("### ⚙️ 買い目フィルター設定")
bet_mode_label = st.sidebar.radio("選定方式", ["期待値(EV)ベース 🆕", "確率ベース(旧)"], index=0)
bet_mode = "ev" if bet_mode_label.startswith("期待値") else "prob"

if bet_mode == "ev":
    ev_threshold = st.sidebar.slider("【購入ライン】期待値EV以上", 1.0, 2.0, 1.2, 0.05)
    min_prob = st.sidebar.slider("【最低予想確率】(%)以上", 0.0, 10.0, 1.0, 0.5)
    st.sidebar.caption("EV=予想確率×オッズ。1.0でトントン。最低確率は『当てにできる確率帯』に絞るための足切り。")
    prob_threshold = 0.0
else:
    prob_threshold = st.sidebar.slider("【購入ライン】確率(%)以上", 0.5, 20.0, 2.5, 0.5)
    ev_threshold = 1.0
    min_prob = 0.0

max_bets = st.sidebar.slider("【上限点数】最大(点)まで", 1, 12, 6, 1)
bet_amount = st.sidebar.number_input("【1点の購入金額】(円)", min_value=100, step=100, value=100)
st.sidebar.caption(f"現在のオッズ並び順: 『{_get_active_ordering()}』（タブ2の検証で自動設定）")
st.sidebar.caption("※この設定は解析・バックテスト・自動購入データすべてに連動します。")

st.title("🚤 v20.0 全艇スコア解析（期待値EVベース）")
st.caption("確率ではなく期待値で買う / 🤖 JSONキューマスター対応版")

tab1, tab2 = st.tabs(["🔍 1レース解析 (当日単発用)", "📊 バックテスト＆JSON生成"])

with tab1:
    col1, col2 = st.columns(2)
    with col1: d_input = st.date_input("日付", value=datetime.now(JST).date())
    with col2: v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
    r_idx = st.selectbox("レース", options=list(range(1, 13)))

    if st.button("🔍 解析 ＆ 買い目生成", type="primary", use_container_width=True):
        dstr = d_input.strftime("%Y%m%d")
        active_ord = _get_active_ordering()
        res = fetch_kyotei24_data(v_idx, r_idx, dstr)
        if res:
            racers, _, _, _ = res
            ranked, _ = rank_all(racers, JCD_NAME[v_idx])
            odds_map = fetch_odds3t(v_idx, r_idx, dstr, ordering=active_ord)

            if not odds_map and bet_mode == "ev":
                st.warning("⚠️ オッズ未取得（締切前/未確定 or 取得失敗）。EVを計算できないため『見送り』です。確率は下表で確認できます。")

            bet_probs, chosen = select_bets(ranked, odds_map, bet_mode, ev_threshold, min_prob, prob_threshold, max_bets)
            buy_bets = [bp["bet"] for bp in chosen]
            st.success("解析完了！")

            st.subheader("🤖 単発レース買い目")
            if buy_bets:
                st.code(",".join(buy_bets), language="text")
                if bet_mode == "ev":
                    inv = len(buy_bets) * bet_amount
                    sum_ev = sum(bp["ev"] for bp in chosen)
                    st.caption(f"投資 {inv:,}円 ／ 理論期待回収 {sum_ev * bet_amount:,.0f}円（採用買い目の合計EV = {sum_ev:.2f}）")
            else:
                st.info("※設定した条件を満たす買い目がないため「見送り」です。")

            st.markdown("---")
            df_disp = []
            for item in ranked:
                racer = item["racer"]; rel = item["rel"]
                df_disp.append({
                    "枠": item["lane"], "選手名": racer.name, "AIスコア": item["score"],
                    "1着率(%)": item["1着率"], "2着率(%)": item["2着率"], "3着率(%)": item["3着率"],
                    "ST": racer.avg_st, "勝率": racer.n_win, "勝率(偏差)": f"{rel['win_dev']:+.2f}"
                })
            st.dataframe(pd.DataFrame(df_disp).set_index("枠"), use_container_width=True)

            st.subheader("🎯 買い目候補（オッズ・EV付き / 上位15点）")
            disp = sorted(bet_probs, key=lambda x: x["ev"], reverse=True) if odds_map else \
                   sorted(bet_probs, key=lambda x: x["prob"], reverse=True)
            rows = []
            for bp in disp[:15]:
                rows.append({
                    "買い目": bp["bet"],
                    "予想確率(%)": bp["prob"],
                    "オッズ": bp["odds"] if bp["odds"] > 0 else "—",
                    "EV(期待値)": bp["ev"] if bp["odds"] > 0 else "—",
                    "判定": "✅ 購入" if bp["bet"] in buy_bets else "見送り"
                })
            st.dataframe(pd.DataFrame(rows).set_index("買い目"), use_container_width=True)

            with st.expander("🌬️ 直前情報（展示タイム・風 / 参考・モデル未使用）"):
                bi = fetch_beforeinfo(v_idx, r_idx, dstr)
                if bi and (bi.get("tenji") or bi.get("weather")):
                    if bi.get("tenji"):
                        st.write("展示タイム： " + " ／ ".join(f"{k}号艇 {v}" for k, v in sorted(bi["tenji"].items())))
                    if bi.get("weather"):
                        st.write("気象： " + " ／ ".join(f"{name} {val}" for name, val in bi["weather"].items()))
                    st.caption("※現行モデルは23特徴量固定のため未使用。特徴量に加えるには再学習が必要です。")
                else:
                    st.caption("直前情報は取得できませんでした（未発表 or 解析失敗）。表示専用機能です。")
        else:
            st.error("出走表が取得できませんでした。")

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1: bt_start = st.date_input("開始日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with col2: bt_end = st.date_input("終了日 ", value=datetime.now(JST).date() - timedelta(days=1))
    with col3: bt_venue_idx = st.selectbox("場を指定", options=[0] + list(JCD_NAME.keys()), format_func=lambda x: "全国（すべて）" if x==0 else JCD_NAME[x])

    mode_txt = f"EV≥{ev_threshold}・最低確率{min_prob}%" if bet_mode == "ev" else f"確率≥{prob_threshold}%"
    st.write(f"※ サイドバー設定（{mode_txt}, 上限{max_bets}点, {bet_amount}円/点, 並び順『{_get_active_ordering()}』）で抽出します。")

    # --- 🆕 オッズ並び順の検証（初回1回） ---
    with st.expander("🔧 オッズ並び順の検証（初回に1回だけ実行を推奨）", expanded=False):
        st.caption("確定済レースの『払戻金』と公式オッズを突き合わせ、正しいセル並び順を自動判定して設定します。EV計算の前提なので最初に必ず実行してください。")
        if st.button("選択中の日付・場で並び順を検証する", use_container_width=True):
            sdays = [(bt_start + timedelta(days=i)).strftime("%Y%m%d") for i in range((bt_end - bt_start).days + 1)]
            vj = bt_venue_idx if bt_venue_idx != 0 else 1
            sample = [(sdays[0], vj, r) for r in range(1, 13)]
            with st.spinner("確定済レースで検証中..."):
                summary = verify_odds_ordering(sample)
            best = min(summary.items(), key=lambda kv: kv[1][0])
            for name, (err, n) in sorted(summary.items(), key=lambda kv: kv[1][0]):
                st.write(f"並び順『{name}』： 平均誤差 **{err*100:.1f}%** （{n}レースで検証）")
            if best[1][1] > 0 and best[1][0] < 0.05:
                st.session_state["odds_ordering"] = best[0]
                st.success(f"✅ 正しい並び順は『{best[0]}』と判定し、設定しました。サイドバー表示が更新されます。")
            else:
                st.error("どの候補も一致しませんでした。公式オッズ表(odds3t)のHTML断片を共有いただければ、新しい並び順を追加します。")

    if st.button("📊 バックテスト実行 ＆ JSON一括生成", type="primary", use_container_width=True):
        active_ord = _get_active_ordering()   # ★メインスレッドで確定し、ワーカーへは引数で渡す
        days = [(bt_start + timedelta(days=i)).strftime("%Y%m%d") for i in range((bt_end - bt_start).days + 1)]
        matches = []
        prog = st.progress(0.0)
        tasks = [(dstr, j, r) for dstr in days for j in ([bt_venue_idx] if bt_venue_idx != 0 else list(range(1, 25))) for r in range(1, 13)]
        st.write(f"全 {len(tasks)} レースを解析中...（各レースでオッズも取得します）")

        def analyze_race(d, j, r):
            res = fetch_kyotei24_data(j, r, d)
            if not res: return None
            racers, lane_to_rank, payoff, has_result = res
            venue_name = JCD_NAME.get(j, "不明")
            ranked, _ = rank_all(racers, venue_name)

            odds_map = fetch_odds3t(j, r, d, ordering=active_ord)
            bet_probs, chosen = select_bets(ranked, odds_map, bet_mode, ev_threshold, min_prob, prob_threshold, max_bets)
            buy_bets = [bp["bet"] for bp in chosen]
            ev_map = {bp["bet"]: bp["ev"] for bp in bet_probs}
            sum_ev = round(sum(bp["ev"] for bp in chosen), 2)

            buy_bets_disp = ",".join([f"{b}(EV{ev_map.get(b, 0.0)})" for b in buy_bets]) if buy_bets else "見"

            if not has_result:
                actual_result = "結果待ち" if buy_bets else "見送り"
                actual_result_disp = actual_result
                hit_str, payoff_disp, hit_amount = "⏳", "-", 0
            else:
                actual_result = ""
                actual_result_disp = ""
                r1 = next((k for k, v in lane_to_rank.items() if str(v) == '1'), None)
                r2 = next((k for k, v in lane_to_rank.items() if str(v) == '2'), None)
                r3 = next((k for k, v in lane_to_rank.items() if str(v) == '3'), None)
                if r1 and r2 and r3:
                    actual_result = f"{r1}-{r2}-{r3}"
                    actual_result_disp = f"{actual_result}(EV{ev_map.get(actual_result, 0.0)})"

                if not buy_bets:
                    hit_str, payoff_disp, hit_amount = "ー", f"({payoff})", 0
                else:
                    hit = actual_result in buy_bets
                    hit_str, payoff_disp, hit_amount = ("🎯" if hit else "❌"), payoff, (payoff if hit else 0)

            return {
                "日付": d, "場": venue_name, "R": r,
                "買い目": ",".join(buy_bets) if buy_bets else "見",
                "買い目(EV)": buy_bets_disp,
                "点数": len(buy_bets),
                "合計EV": sum_ev,
                "結果": actual_result,
                "結果(EV)": actual_result_disp,
                "的中": hit_str, "払戻金": payoff_disp, "_hit_amount": hit_amount
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
            bet_races = [m for m in matches if m["点数"] > 0 and m["的中"] in ["🎯", "❌", "⏳"]]
            hits = [m for m in bet_races if m["的中"] == "🎯"]
            settled = [m for m in bet_races if m["的中"] != "⏳"]

            if bet_races:
                total_invest = sum(m["点数"] for m in settled) * bet_amount
                total_return = sum(m["_hit_amount"] for m in settled) * (bet_amount / 100)
                hit_rate = len(hits) / len(settled) * 100 if settled else 0
                ret_rate = total_return / total_invest * 100 if total_invest > 0 else 0
                avg_ev = (sum(m["合計EV"] for m in settled) / sum(m["点数"] for m in settled)) if sum(m["点数"] for m in settled) > 0 else 0

                st.success(f"🔥 勝負対象レース: {len(bet_races)}件 (見送り: {len([m for m in matches if m['点数']==0])}件)")
                if total_invest > 0:
                    st.info(f"💰 **総投資**: {total_invest:,.0f}円 ／ **総回収**: {total_return:,.0f}円 （回収率: {ret_rate:.1f}%）")
                    st.caption(f"参考：採用買い目の平均EV = {avg_ev:.2f}。理論上はこれが回収率(={ret_rate/100:.2f})に近づくはず。"
                               f"大きく下振れしていればモデル確率が過大評価（要キャリブレーション）の疑い。")

                disp_cols = ["日付", "場", "R", "買い目(EV)", "結果(EV)", "的中", "払戻金"]
                st.dataframe(df_bt[disp_cols], use_container_width=True)

                # --- JSON生成（自動購入用） ---
                auto_bet_queue = []
                for index, row in df_bt.iterrows():
                    if row["点数"] > 0 and row["買い目"] != "見":
                        bets = row["買い目"].split(",")
                        for bet in bets:
                            try:
                                pattern = [int(x) for x in bet.split("-")]
                                queue_item = {
                                    "venue": row["場"],
                                    "race": str(row["R"]),
                                    "pattern": pattern,
                                    "amount": str(bet_amount)
                                }
                                auto_bet_queue.append(queue_item)
                            except:
                                pass

                if auto_bet_queue:
                    json_string = json.dumps(auto_bet_queue, ensure_ascii=False, indent=4)
                    st.markdown("---")
                    st.subheader("🤖 全自動購入用データ (キューマスター専用)")
                    st.caption("右上のコピーボタンを押し、テレボート画面のダッシュボードに貼り付けてください。")
                    st.code(json_string, language="json")
        else:
            st.error("対象期間のデータが取得できませんでした。")
