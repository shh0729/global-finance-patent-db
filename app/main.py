import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 基础配置
# =========================================================
st.set_page_config(
    page_title="金融专利数据库",
    page_icon="📊",
    layout="wide",
)

DB_PATH = Path(__file__).resolve().parent.parent / "global_finance_patents.db"

COUNTRY_NAMES = {
    "CN": "中国",
    "US": "美国",
    "WO": "世界知识产权组织（WIPO）",
    "EP": "欧洲专利局（EPO）",
    "KR": "韩国",
    "JP": "日本",
    "TW": "中国台湾",
    "AU": "澳大利亚",
    "HK": "中国香港",
    "GB": "英国",
    "SG": "新加坡",
    "CA": "加拿大",
    "PH": "菲律宾",
    "IT": "意大利",
    "FI": "芬兰",
}


def country_name(code):
    if pd.isna(code):
        return "未知"
    return COUNTRY_NAMES.get(str(code), f"其他申请局（{code}）")


# =========================================================
# 数据读取：兼容 title_zh 字段暂时不存在的情况
# =========================================================
@st.cache_data
def load_data(db_path, db_mtime):
    if not Path(db_path).exists():
        raise FileNotFoundError(f"找不到数据库文件：{db_path}")

    conn = sqlite3.connect(db_path)
    try:
        text_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(patent_text)").fetchall()
        }
        title_zh_sql = "title_zh" if "title_zh" in text_columns else "'' AS title_zh"

        applications = pd.read_sql_query(
            """
            SELECT
                a.appln_id,
                a.docdb_family_id,
                a.appln_auth,
                a.appln_nr,
                a.appln_kind,
                a.appln_filing_date,
                a.appln_filing_year,
                f.family_size,
                f.earliest_filing_year
            FROM patent_application a
            LEFT JOIN patent_family f
                ON a.docdb_family_id = f.docdb_family_id
            """,
            conn,
        )

        texts = pd.read_sql_query(
            f"""
            SELECT appln_id, language, title, {title_zh_sql}, abstract
            FROM patent_text
            """,
            conn,
        )

        classifications = pd.read_sql_query(
            """
            SELECT appln_id, cpc_class_symbol
            FROM patent_classification
            """,
            conn,
        )

        publications = pd.read_sql_query(
            """
            SELECT
                appln_id, publn_auth, publn_nr, publn_kind,
                publn_date, publication_url
            FROM patent_publication
            """,
            conn,
        )

        domains = pd.read_sql_query(
            """
            SELECT docdb_family_id, domain_code, domain_name
            FROM finance_family
            """,
            conn,
        )
    finally:
        conn.close()

    return applications, texts, classifications, publications, domains


try:
    db_mtime = Path(DB_PATH).stat().st_mtime
    applications, texts, classifications, publications, domains = load_data(
        str(DB_PATH), db_mtime
    )
except Exception as exc:
    st.error(f"数据库读取失败：{exc}")
    st.stop()


# =========================================================
# 文本、公开链接和基础数据整理
# =========================================================
texts["language"] = texts["language"].fillna("").astype(str)
english_texts = texts[texts["language"].str.lower().eq("en")].copy()
english_texts = english_texts.drop_duplicates(subset=["appln_id"], keep="first")

data = applications.merge(
    english_texts[["appln_id", "title", "title_zh", "abstract"]],
    on="appln_id",
    how="left",
)

for col, fallback in [
    ("title", "暂无英文标题"),
    ("title_zh", ""),
    ("abstract", "暂无摘要"),
]:
    data[col] = data[col].fillna("").astype(str)
    if fallback and col == "title":
        data[col] = data[col].replace("", fallback)
    if fallback and col == "abstract":
        data[col] = data[col].replace("", fallback)

data["display_title"] = data["title_zh"].where(
    data["title_zh"].str.strip().ne(""),
    data["title"],
)
data["appln_filing_year"] = pd.to_numeric(
    data["appln_filing_year"], errors="coerce"
)

publications["publication_url"] = publications["publication_url"].fillna("").astype(str)
valid_publications = publications[
    publications["publication_url"].str.startswith(("http://", "https://"))
].copy()

# 每件申请汇总公开链接，检索结果表格可直接点击
if not valid_publications.empty:
    link_rows = (
        valid_publications.sort_values(["appln_id", "publn_date"], na_position="last")
        .groupby("appln_id")["publication_url"]
        .agg(list)
        .reset_index(name="publication_urls")
    )
    link_rows["原文链接"] = link_rows["publication_urls"].apply(
        lambda urls: urls[0] if urls else ""
    )
    link_rows["公开记录数"] = link_rows["publication_urls"].apply(len)
else:
    link_rows = pd.DataFrame(columns=["appln_id", "publication_urls", "原文链接", "公开记录数"])

data = data.merge(
    link_rows[["appln_id", "原文链接", "公开记录数"]],
    on="appln_id",
    how="left",
)
data["原文链接"] = data["原文链接"].fillna("")
data["公开记录数"] = data["公开记录数"].fillna(0).astype(int)

# 每件申请拼接 CPC，用于技术分类识别
if not classifications.empty:
    cpc_by_app = (
        classifications.groupby("appln_id")["cpc_class_symbol"]
        .agg(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index(name="cpc_text")
    )
    data = data.merge(cpc_by_app, on="appln_id", how="left")
else:
    data["cpc_text"] = ""
data["cpc_text"] = data["cpc_text"].fillna("")


# =========================================================
# 可切换的分类方案
# =========================================================
BUSINESS_RULES = {
    "支付与结算": [
        "支付", "结算", "汇款", "转账", "电子钱包", "移动支付",
        "payment", "payments", "settlement", "remittance", "wallet",
        "money transfer", "fund transfer",
    ],
    "银行与账户服务": [
        "银行", "账户", "存款", "取款", "bank", "banking",
        "account management", "deposit", "withdrawal",
    ],
    "信贷与金融风控": [
        "信贷", "贷款", "借款", "信用评估", "征信", "风险评估",
        "credit", "loan", "lending", "credit scoring", "risk assessment",
        "fraud detection",
    ],
    "保险科技": [
        "保险", "保单", "理赔", "insurance", "policy", "claim",
        "underwriting", "premium",
    ],
    "投资与证券": [
        "投资", "证券", "股票", "交易", "资产配置", "investment",
        "securities", "stock", "trading", "portfolio", "asset allocation",
    ],
    "税务科技": [
        "税务", "纳税", "税收", "tax", "taxation", "tax payment",
        "tax return",
    ],
    "金融数据与客户服务": [
        "客户", "用户画像", "金融数据", "financial data", "customer",
        "client", "recommendation", "personal finance",
    ],
}

TECH_RULES = {
    "人工智能与机器学习": [
        "人工智能", "机器学习", "深度学习", "神经网络", "智能识别",
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "predictive model",
    ],
    "区块链与分布式账本": [
        "区块链", "分布式账本", "智能合约", "blockchain",
        "distributed ledger", "smart contract",
    ],
    "大数据与数据分析": [
        "大数据", "数据挖掘", "数据分析", "big data", "data mining",
        "data analysis", "analytics",
    ],
    "云计算与平台技术": [
        "云计算", "云平台", "cloud computing", "cloud platform",
        "server platform",
    ],
    "密码学与信息安全": [
        "加密", "密码", "身份认证", "隐私保护", "信息安全",
        "encryption", "cryptography", "authentication", "privacy",
        "cybersecurity", "security protocol",
    ],
    "移动互联网与终端": [
        "移动终端", "移动设备", "手机", "mobile device",
        "mobile terminal", "smartphone", "wireless communication",
    ],
    "其他数字技术": [
        "数字身份", "物联网", "分布式", "digital identity",
        "internet of things", "iot", "distributed system",
    ],
}


def classify_by_rules(text, rules):
    text = str(text).lower()
    matches = []
    for category, keywords in rules.items():
        if any(str(keyword).lower() in text for keyword in keywords):
            matches.append(category)
    return matches or ["其他/待识别"]


def parse_custom_rules(raw_text):
    rules = {}
    for line in raw_text.splitlines():
        if ":" not in line and "：" not in line:
            continue
        separator = ":" if ":" in line else "："
        category, words = line.split(separator, 1)
        category = category.strip()
        keywords = [x.strip() for x in words.replace("，", ",").split(",") if x.strip()]
        if category and keywords:
            rules[category] = keywords
    return rules


st.title("🌐 金融专利数据库")
st.caption("Financial Patent Database")
st.info(
    "数据来源：EPO PATSTAT TEST A 样本。当前数据库约包含921件专利申请、"
    "494个专利族和1478条公开记录。该样本不代表全球专利全量数据。"
)

st.sidebar.title("数据库导航")
page = st.sidebar.radio(
    "选择功能",
    ["数据总览", "专利检索", "统计分析", "专利详情"],
)
st.sidebar.divider()
st.sidebar.subheader("技术领域分类方案")

scheme_names = {
    "现有分类（原始数据库）": "legacy",
    "金融业务分类": "business",
    "技术类型分类": "technology",
    "自定义分类规则": "custom",
}
selected_scheme_label = st.sidebar.selectbox(
    "选择分类方案",
    list(scheme_names.keys()),
)
selected_scheme = scheme_names[selected_scheme_label]

custom_rules = {}
if selected_scheme == "custom":
    st.sidebar.caption("每行一种类别，格式：类别名称：关键词1,关键词2")
    custom_text = st.sidebar.text_area(
        "自定义分类规则",
        value=(
            "支付与结算：支付,payment,settlement,汇款\n"
            "银行与信贷：bank,banking,loan,credit,银行,贷款\n"
            "保险科技：insurance,claim,保险,理赔\n"
            "投资与证券：investment,securities,trading,投资,证券"
        ),
        height=150,
    )
    custom_rules = parse_custom_rules(custom_text)
    if not custom_rules:
        st.sidebar.warning("尚未识别到有效规则，请按提示填写。")

# 生成当前分类标签。原始分类保留，不覆盖数据库中的原始字段。
legacy_map = (
    domains.groupby("docdb_family_id")["domain_name"]
    .agg(lambda x: list(dict.fromkeys(x.dropna().astype(str))))
    .to_dict()
)

def get_categories(row):
    if selected_scheme == "legacy":
        values = legacy_map.get(row["docdb_family_id"], [])
        return values or ["原始分类未标注"]
    text = " ".join(
        [
            str(row.get("title", "")),
            str(row.get("title_zh", "")),
            str(row.get("abstract", "")),
            str(row.get("cpc_text", "")),
        ]
    )
    if selected_scheme == "business":
        return classify_by_rules(text, BUSINESS_RULES)
    if selected_scheme == "technology":
        return classify_by_rules(text, TECH_RULES)
    if selected_scheme == "custom":
        return classify_by_rules(text, custom_rules) if custom_rules else ["未设置规则"]
    return ["其他/待识别"]

data["技术领域分类"] = data.apply(get_categories, axis=1)
data["分类方案"] = selected_scheme_label

st.sidebar.caption(
    "说明：业务分类、技术分类和自定义分类依据标题、摘要及CPC文本关键词自动匹配；"
    "属于辅助识别结果，建议抽样核验。"
)


# =========================================================
# 通用筛选
# =========================================================
def get_filtered_data():
    df = data.copy()
    st.sidebar.divider()
    st.sidebar.subheader("数据筛选")

    years = sorted(df["appln_filing_year"].dropna().astype(int).unique())
    if years:
        year_range = st.sidebar.slider(
            "申请年份范围",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years))),
        )
        df = df[
            df["appln_filing_year"].between(year_range[0], year_range[1])
        ]

    country_codes = sorted(df["appln_auth"].dropna().astype(str).unique())
    selected_countries = st.sidebar.multiselect(
        "申请国家/地区或专利局",
        country_codes,
        default=[],
        format_func=country_name,
    )
    if selected_countries:
        df = df[df["appln_auth"].isin(selected_countries)]

    category_options = sorted(
        {category for categories in df["技术领域分类"] for category in categories}
    )
    selected_categories = st.sidebar.multiselect(
        "技术领域分类筛选",
        category_options,
        default=[],
    )
    if selected_categories:
        df = df[
            df["技术领域分类"].apply(
                lambda categories: any(x in categories for x in selected_categories)
            )
        ]
    return df


filtered = get_filtered_data()


# =========================================================
# 页面一：数据总览
# =========================================================
if page == "数据总览":
    st.header("📊 金融专利数据总览")

    total_applications = filtered["appln_id"].nunique()
    total_families = filtered["docdb_family_id"].nunique()
    total_publications = int(filtered["公开记录数"].sum())
    total_offices = filtered["appln_auth"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("专利申请数量", f"{total_applications:,}")
    c2.metric("专利族数量", f"{total_families:,}")
    c3.metric("公开记录数量", f"{total_publications:,}")
    c4.metric("申请局数量", total_offices)

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("年度专利申请趋势")
        yearly = (
            filtered.dropna(subset=["appln_filing_year"])
            .groupby("appln_filing_year")["appln_id"]
            .nunique()
            .reset_index(name="年度新增申请量")
            .sort_values("appln_filing_year")
        )
        if not yearly.empty:
            fig = px.line(
                yearly, x="appln_filing_year", y="年度新增申请量",
                markers=True, title="各年度新增专利申请数量",
            )
            fig.update_layout(xaxis_title="申请年份", yaxis_title="申请件数")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("申请局申请数量")
        country = (
            filtered.groupby("appln_auth")["appln_id"]
            .nunique()
            .reset_index(name="申请数量")
            .sort_values("申请数量", ascending=True)
            .tail(10)
        )
        if not country.empty:
            country["申请局/国家地区"] = country["appln_auth"].apply(country_name)
            fig = px.bar(
                country, x="申请数量", y="申请局/国家地区",
                orientation="h", text="申请数量",
                title="申请数量排名前十",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"技术领域结构：{selected_scheme_label}")
    category_rows = []
    for _, row in filtered.iterrows():
        for category in row["技术领域分类"]:
            category_rows.append(
                {"技术领域": category, "docdb_family_id": row["docdb_family_id"]}
            )
    category_df = pd.DataFrame(category_rows)
    if not category_df.empty:
        category_count = (
            category_df.groupby("技术领域")["docdb_family_id"]
            .nunique()
            .reset_index(name="涉及专利族数")
            .sort_values("涉及专利族数", ascending=False)
        )
        fig = px.pie(
            category_count, names="技术领域", values="涉及专利族数",
            hole=0.4, title="按当前分类方案统计的专利族分布",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(category_count, use_container_width=True, hide_index=True)
    else:
        st.warning("当前筛选条件下没有可统计的分类数据。")


# =========================================================
# 页面二：专利检索
# =========================================================
elif page == "专利检索":
    st.header("🔎 金融专利检索")
    keyword = st.text_input(
        "关键词检索",
        placeholder="输入中文或英文标题、摘要、申请编号",
    )

    cpc_options = sorted(
        classifications["cpc_class_symbol"].dropna().astype(str).unique().tolist()
    )
    selected_cpc = st.selectbox("CPC技术分类", ["全部分类"] + cpc_options)

    result = filtered.copy()
    if keyword:
        term = keyword.lower()
        mask = (
            result["title"].str.lower().str.contains(term, na=False, regex=False)
            | result["title_zh"].str.lower().str.contains(term, na=False, regex=False)
            | result["abstract"].str.lower().str.contains(term, na=False, regex=False)
            | result["appln_id"].astype(str).str.contains(term, na=False, regex=False)
            | result["appln_nr"].fillna("").astype(str).str.lower().str.contains(
                term, na=False, regex=False
            )
        )
        result = result[mask]

    if selected_cpc != "全部分类":
        matched_ids = classifications[
            classifications["cpc_class_symbol"].astype(str).eq(selected_cpc)
        ]["appln_id"].unique()
        result = result[result["appln_id"].isin(matched_ids)]

    st.write(f"检索结果：**{result['appln_id'].nunique():,}** 件专利申请")
    if not result.empty:
        display = result[
            [
                "appln_id", "display_title", "appln_auth",
                "appln_filing_year", "docdb_family_id",
                "公开记录数", "原文链接",
            ]
        ].copy()
        display["appln_auth"] = display["appln_auth"].apply(country_name)
        display.columns = [
            "申请编号", "专利标题", "申请国家/地区",
            "申请年份", "专利族编号", "公开记录数", "专利原文",
        ]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "专利原文": st.column_config.LinkColumn(
                    "专利原文",
                    display_text="打开原文 ↗",
                )
            },
        )
        st.caption("点击“打开原文 ↗”即可跳转 Google Patents；若无链接，该单元格为空。")
        st.download_button(
            "下载当前检索结果 CSV",
            result.to_csv(index=False).encode("utf-8-sig"),
            file_name="finance_patent_search.csv",
            mime="text/csv",
        )
    else:
        st.warning("没有找到符合条件的专利。")


# =========================================================
# 页面三：统计分析——按单个申请局/国家地区逐年统计
# =========================================================
elif page == "统计分析":
    st.header("📈 国家/地区年度统计分析")
    st.caption(
        "本页按单个申请局分别统计，不是把所有国家合并成一条全球年度趋势。"
        "累计统计固定从2000年起算；年份范围只控制表格和图表展示范围。"
    )

    full = data.dropna(subset=["appln_filing_year"]).copy()
    full["appln_filing_year"] = full["appln_filing_year"].astype(int)
    available_countries = sorted(full["appln_auth"].dropna().astype(str).unique())

    col1, col2 = st.columns(2)
    with col1:
        chosen_country = st.selectbox(
            "选择一个国家/地区或申请局",
            available_countries,
            format_func=country_name,
        )
    with col2:
        metric_mode = st.radio(
            "统计口径",
            ["申请件数", "专利族件数"],
            horizontal=True,
        )

    min_data_year = max(2000, int(full["appln_filing_year"].min()))
    max_data_year = int(full["appln_filing_year"].max())
    if min_data_year > max_data_year:
        st.warning("数据库中没有2000年及以后的有效申请年份。")
        st.stop()

    display_range = st.slider(
        "展示年份范围（累计量仍从2000年开始计算）",
        min_value=min_data_year,
        max_value=max_data_year,
        value=(min_data_year, max_data_year),
    )

    years_all = list(range(2000, max_data_year + 1))
    years_display = list(range(display_range[0], display_range[1] + 1))

    # 全样本年度分母
    if metric_mode == "申请件数":
        global_annual = (
            full.groupby("appln_filing_year")["appln_id"].nunique()
            .reindex(years_all, fill_value=0)
        )
        country_annual = (
            full[full["appln_auth"].astype(str).eq(chosen_country)]
            .groupby("appln_filing_year")["appln_id"].nunique()
            .reindex(years_all, fill_value=0)
        )
        global_cumulative = global_annual.cumsum()
        country_cumulative = country_annual.cumsum()
        count_label = "年度新增申请件数"
        cumulative_label = "累计申请件数"
    else:
        global_annual = (
            full.groupby("appln_filing_year")["docdb_family_id"].nunique()
            .reindex(years_all, fill_value=0)
        )
        country_subset = full[full["appln_auth"].astype(str).eq(chosen_country)]
        country_annual = (
            country_subset.groupby("appln_filing_year")["docdb_family_id"].nunique()
            .reindex(years_all, fill_value=0)
        )
        # 专利族累计量按截至该年去重，不对年度族数简单求和
        country_cumulative = pd.Series(
            [
                country_subset.loc[
                    country_subset["appln_filing_year"] <= year,
                    "docdb_family_id",
                ].nunique()
                for year in years_all
            ],
            index=years_all,
        )
        global_cumulative = pd.Series(
            [
                full.loc[
                    full["appln_filing_year"] <= year,
                    "docdb_family_id",
                ].nunique()
                for year in years_all
            ],
            index=years_all,
        )
        count_label = "当年涉及专利族数"
        cumulative_label = "截至当年累计专利族数"

    previous = country_annual.shift(1)
    yoy = ((country_annual / previous - 1) * 100).where(previous > 0)

    table = pd.DataFrame(
        {
            "申请年份": years_all,
            count_label: country_annual.values,
            "上一年新增量": previous.values,
            "同比增长率（%）": yoy.round(2).values,
            "样本当年总量": global_annual.values,
            "当年占样本比例（%）": [
                round(country_annual.loc[y] / global_annual.loc[y] * 100, 2)
                if global_annual.loc[y] > 0 else None
                for y in years_all
            ],
            cumulative_label: country_cumulative.values,
            "样本累计总量": global_cumulative.values,
            "累计占样本比例（%）": [
                round(country_cumulative.loc[y] / global_cumulative.loc[y] * 100, 2)
                if global_cumulative.loc[y] > 0 else None
                for y in years_all
            ],
        }
    )
    table = table[table["申请年份"].isin(years_display)].copy()

    st.subheader(f"{country_name(chosen_country)}：{metric_mode}")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "下载该国家年度统计 CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{chosen_country}_{metric_mode}_annual_statistics.csv",
        mime="text/csv",
    )

    chart_data = table.melt(
        id_vars="申请年份",
        value_vars=[count_label, cumulative_label],
        var_name="指标",
        value_name="数量",
    )
    fig = px.line(
        chart_data,
        x="申请年份",
        y="数量",
        color="指标",
        markers=True,
        title=f"{country_name(chosen_country)}年度新增量与累计量",
    )
    st.plotly_chart(fig, use_container_width=True)

    yoy_chart = table[["申请年份", "同比增长率（%）"]].dropna()
    if not yoy_chart.empty:
        fig_yoy = px.bar(
            yoy_chart,
            x="申请年份",
            y="同比增长率（%）",
            title=f"{country_name(chosen_country)}年度新增量同比增长率",
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    st.caption(
        "口径说明：申请件数按申请编号去重；专利族件数按DOCDB专利族去重。"
        "国家/申请局的累计专利族数按截至当年出现过的不同专利族计算，"
        "避免同一专利族跨年度申请时重复累加。"
        "占比的分母是当前PATSTAT TEST A数据库样本，不代表真实全球总量。"
        "WO和EP分别是国际申请体系及欧洲专利局口径，并非主权国家。"
    )


# =========================================================
# 页面四：专利详情
# =========================================================
elif page == "专利详情":
    st.header("📄 专利详情")
    available_ids = sorted(filtered["appln_id"].unique().tolist())

    if not available_ids:
        st.warning("当前筛选条件下没有可查看的专利。")
    else:
        selected_id = st.selectbox("选择专利申请编号", available_ids)
        patent = data[data["appln_id"] == selected_id].iloc[0]

        st.subheader(patent["display_title"])
        c1, c2, c3 = st.columns(3)
        c1.metric("申请编号", str(patent["appln_id"]))
        c2.metric("申请国家/地区", country_name(patent["appln_auth"]))
        c3.metric("申请年份", str(patent["appln_filing_year"]))

        st.write("**英文标题**")
        st.write(patent["title"])
        if patent["title_zh"].strip():
            st.write("**中文标题**")
            st.write(patent["title_zh"])
        st.write("**专利摘要**")
        st.write(patent["abstract"])

        st.divider()
        st.write("**技术领域分类**")
        st.write("；".join(patent["技术领域分类"]))
        st.caption(f"当前分类方案：{selected_scheme_label}")

        st.write("**CPC分类**")
        patent_cpc = classifications[classifications["appln_id"] == selected_id]
        if not patent_cpc.empty:
            st.dataframe(patent_cpc, use_container_width=True, hide_index=True)
        else:
            st.write("暂无CPC分类信息")

        st.divider()
        st.subheader("专利公开记录与原文链接")
        patent_publications = publications[
            publications["appln_id"] == selected_id
        ].copy()

        if not patent_publications.empty:
            patent_publications = patent_publications.reset_index(drop=True)
            for idx, row in patent_publications.iterrows():
                pub_number = " ".join(
                    str(x) for x in [row.get("publn_auth"), row.get("publn_nr"), row.get("publn_kind")]
                    if pd.notna(x) and str(x).strip()
                )
                pub_date = row.get("publn_date", "")
                url = str(row.get("publication_url", "") or "")
                left, right = st.columns([4, 1])
                with left:
                    st.write(f"**公开记录 {idx + 1}：{pub_number}**")
                    st.caption(f"公开日期：{pub_date}")
                with right:
                    if url.startswith(("http://", "https://")):
                        st.link_button("打开专利原文 ↗", url, use_container_width=True)
                    else:
                        st.caption("暂无有效链接")
                st.divider()
        else:
            st.write("暂无公开记录")
