# 先安装依赖（首次运行需执行）
import subprocess
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st

def install_deps():
    """安装或升级所需的依赖库"""
    required_packages = ['streamlit>=1.28.0', 'pandas', 'plotly', 'openpyxl', 'numpy']
    try:
        import pkg_resources
        installed = {p.key for p in pkg_resources.working_set}
        # 简化逻辑，直接尝试升级或安装，确保是最新兼容版本
        print(f"正在检查并安装/升级依赖库: {', '.join(required_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *required_packages])
        print("依赖库安装/升级完成。")
    except Exception as e:
        print(f"自动安装依赖失败: {e}")
        print("请手动安装以下库: " + ", ".join(required_packages))

# 首次运行或遇到导入错误时，尝试安装依赖
try:
    # 检查 streamlit 版本
    from importlib.metadata import version
    st_version = version('streamlit')
    print(f"当前 Streamlit 版本: {st_version}")
    # 如果版本过低，触发重新安装
    if tuple(map(int, st_version.split('.'))) < (1, 28, 0):
        print("Streamlit 版本过低，需要升级...")
        raise ImportError("Streamlit version too old")

except (ImportError, Exception):
    print("检测到缺失依赖或版本不兼容，正在尝试自动安装...")
    install_deps()
    # 安装后再次导入
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

# ---------------------- 全局配置 ----------------------
st.set_page_config(
    page_title="网易云歌单数据分析工具",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* 卡片样式 */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* 数据指标卡片 */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# 颜色配置
COLOR_PALETTE = {
    'primary': '#1DB954',      # 网易云红色
    'secondary': '#FF6B6B',    # 辅助红色
    'accent': '#4ECDC4',       # 蓝绿色
    'background': '#F8F9FA',   # 背景色
    'text': '#333333',         # 文本色
    'light_text': '#666666'    # 浅色文本
}

TYPE_LIST = ['流行', '热血', '00后', '华语', '伤感', '夜晚', '治愈', '放松', '感动', '安静', '民谣', '孤独', '浪漫']
DATA_DIR = Path(__file__).parent

# ---------------------- 数据加载与预处理模块 ----------------------
def load_and_preprocess_all_data():
    all_data = []
    found_files = []
    skipped_files = []

    for cat in TYPE_LIST:
        file_path = DATA_DIR / f"{cat}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')
                
                if df.empty:
                    skipped_files.append(f"{cat}.csv (文件为空)")
                    continue

                required_columns = ['名称', '创建日期', '播放次数', '收藏量', '转发量', '评论数', '歌单长度', 'tag1']
                if not all(col in df.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    skipped_files.append(f"{cat}.csv (缺少列: {', '.join(missing_cols)})")
                    continue

                df['分类'] = cat.strip()
                all_data.append(df)
                found_files.append(cat)
            except Exception as e:
                skipped_files.append(f"{cat}.csv (读取错误: {str(e)})")
        else:
            skipped_files.append(f"{cat}.csv (文件不存在)")

    if not all_data:
        st.error("❌ 未成功加载任何数据，请检查CSV文件是否存在且格式正确。")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # ---------------------- 增强去重逻辑 ----------------------
    # 【关键】多列联合去重：名称+分类+创建日期+播放次数完全一致才视为重复
    duplicate_cols = ['名称', '分类', '创建日期']  
    before_count = len(combined_df)  # 去重前数量
    combined_df = combined_df.drop_duplicates(subset=duplicate_cols, keep='first')  # 保留第一条
    after_count = len(combined_df)   # 去重后数量
    # 显示去重结果（直观看到效果）
    st.info(f"🔍 数据去重完成：共移除 {before_count - after_count} 条重复歌单（去重依据：{', '.join(duplicate_cols)}）")

    # 数据预处理（原逻辑保留）
    combined_df['创建日期'] = pd.to_datetime(combined_df['创建日期'], errors='coerce')
    
    numeric_cols = ['播放次数', '收藏量', '转发量', '评论数', '歌单长度']
    for col in numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)
    
    combined_df['tag1'] = combined_df['tag1'].str.replace('nan', '').str.strip()
    
    # 计算衍生指标
    combined_df['收藏播放比'] = (combined_df['收藏量'] / combined_df['播放次数'] * 100).round(4)
    combined_df['评论播放比'] = (combined_df['评论数'] / combined_df['播放次数'] * 100).round(4)
    combined_df['创建月份'] = combined_df['创建日期'].dt.to_period('M')
    
    # 加载总结
    st.success(f"✅ 成功加载 {len(found_files)} / {len(TYPE_LIST)} 个分类的数据。")
    if found_files:
        st.markdown(f"📊 **已加载分类**: {', '.join(found_files)}")
        if not pd.isna(combined_df['创建日期'].min()):
            st.markdown(f"📊 数据时间范围：{combined_df['创建日期'].min().strftime('%Y-%m-%d')} 至 {combined_df['创建日期'].max().strftime('%Y-%m-%d')}")
    
    if skipped_files:
        with st.expander("⚠️ 查看被跳过的文件", expanded=False):
            for reason in skipped_files:
                st.write(reason)
    
    return combined_df

# ---------------------- 数据概览卡片 ----------------------
def display_data_overview(df):
    """显示数据概览指标卡片"""
    st.subheader("📈 数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #1DB954;">总歌单数量</h4>
            <p style="font-size: 24px; font-weight: bold;">{:,}</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #FF6B6B;">总播放次数</h4>
            <p style="font-size: 24px; font-weight: bold;">{:,}</p>
        </div>
        """.format(df['播放次数'].sum()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #4ECDC4;">总收藏量</h4>
            <p style="font-size: 24px; font-weight: bold;">{:,}</p>
        </div>
        """.format(df['收藏量'].sum()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #9B59B6;">平均歌单长度</h4>
            <p style="font-size: 24px; font-weight: bold;">{:.1f}</p>
        </div>
        """.format(df['歌单长度'].mean()), unsafe_allow_html=True)

# ---------------------- 高级可视化模块 ----------------------
def plot_advanced_visualizations(df):
    if df.empty:
        return
    
    st.subheader("🎯 深度数据分析")
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(['分类分析', '时间趋势', '相关性分析', '高级洞察'])
    
    # Tab 1: 分类分析
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # 各分类歌单数量
            cat_counts = df['分类'].value_counts()
            fig = px.bar(
                x=cat_counts.index,
                y=cat_counts.values,
                title='各分类歌单数量分布',
                labels={'x': '分类', 'y': '歌单数量'},
                color=cat_counts.values,
                color_continuous_scale='Reds',
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 各分类平均播放量
            avg_play = df.groupby('分类')['播放次数'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=avg_play.index,
                y=avg_play.values,
                title='各分类平均播放量',
                labels={'x': '分类', 'y': '平均播放次数'},
                color=avg_play.values,
                color_continuous_scale='Blues',
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # 各分类综合指标雷达图
        st.markdown("### 各分类综合表现对比")
        top_categories = df['分类'].value_counts().head(6).index
        cat_metrics = df[df['分类'].isin(top_categories)].groupby('分类').agg({
            '播放次数': 'mean',
            '收藏量': 'mean',
            '评论数': 'mean',
            '歌单长度': 'mean'
        }).reset_index()
        
        # 数据标准化
        for col in ['播放次数', '收藏量', '评论数', '歌单长度']:
            cat_metrics[col] = (cat_metrics[col] - cat_metrics[col].min()) / (cat_metrics[col].max() - cat_metrics[col].min())
        
        fig = go.Figure()
        for _, row in cat_metrics.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['播放次数'], row['收藏量'], row['评论数'], row['歌单长度']],
                theta=['播放次数', '收藏量', '评论数', '歌单长度'],
                name=row['分类']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: 时间趋势
    with tab2:
        # 按月份统计歌单创建数量
        monthly_trend = df.groupby('创建月份').size().reset_index(name='歌单数量')
        monthly_trend['创建月份'] = monthly_trend['创建月份'].astype(str)
        
        fig = px.line(
            monthly_trend,
            x='创建月份',
            y='歌单数量',
            title='歌单创建时间趋势',
            labels={'创建月份': '月份', '歌单数量': '新增歌单数量'},
            template='plotly_white',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 近6个月各分类歌单增长情况
        recent_months = df['创建月份'].unique()[-6:] if len(df['创建月份'].unique()) >=6 else df['创建月份'].unique()
        recent_data = df[df['创建月份'].isin(recent_months)]
        
        if len(recent_data) > 0:
            monthly_cat = recent_data.groupby(['创建月份', '分类']).size().reset_index(name='歌单数量')
            monthly_cat['创建月份'] = monthly_cat['创建月份'].astype(str)
            
            fig = px.area(
                monthly_cat,
                x='创建月份',
                y='歌单数量',
                color='分类',
                title='近6个月各分类歌单增长趋势',
                labels={'创建月份': '月份', '歌单数量': '歌单数量'},
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: 相关性分析
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # 播放量vs收藏量散点图
            fig = px.scatter(
                df,
                x='播放次数',
                y='收藏量',
                color='分类',
                size='歌单长度',
                hover_data=['名称', '创建日期'],
                title='播放量 vs 收藏量',
                labels={'播放次数': '播放次数', '收藏量': '收藏量'},
                opacity=0.7,
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 播放量vs评论数散点图
            fig = px.scatter(
                df,
                x='播放次数',
                y='评论数',
                color='分类',
                size='收藏量',
                hover_data=['名称', '创建日期'],
                title='播放量 vs 评论数',
                labels={'播放次数': '播放次数', '评论数': '评论数'},
                opacity=0.7,
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # 数值特征相关性热力图
        numeric_features = ['播放次数', '收藏量', '转发量', '评论数', '歌单长度', '收藏播放比', '评论播放比']
        corr_matrix = df[numeric_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='特征相关性热力图',
            labels=dict(color='相关系数'),
            x=numeric_features,
            y=numeric_features,
            color_continuous_scale='RdBu_r',
            template='plotly_white'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: 高级洞察
    with tab4:
        # Top 10 高收藏播放比歌单
        st.markdown("### Top 10 高收藏率歌单")
        # 过滤掉播放次数为0的歌单，避免除以零错误
        # 先按收藏播放比降序排序，再按名称去重（保留第一条，即收藏率最高的）
        high_fav_ratio_df = (
            df[df['播放次数'] > 1000]
            .sort_values('收藏播放比', ascending=False)  # 按收藏率降序
            .drop_duplicates(subset='名称', keep='first')  # 按名称去重，保留第一条（收藏率最高）
            .nlargest(10, '收藏播放比')  # 取Top 10
            [['名称', '分类', '播放次数', '收藏量', '收藏播放比', '创建日期']]
        )
        
        fig = px.bar(
            high_fav_ratio_df,
            x='名称',
            y='收藏播放比',
            color='分类',
            title='收藏率最高的10个歌单 (收藏量/播放量%)',
            labels={'名称': '歌单名称', '收藏播放比': '收藏率(%)'},
            template='plotly_white',
            hover_data=['播放次数', '收藏量', '创建日期']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 歌单长度分布
        st.markdown("### 歌单长度分布")
        fig = px.histogram(
            df,
            x='歌单长度',
            nbins=30,
            title='歌单长度分布',
            labels={'歌单长度': '歌曲数量', 'count': '歌单数量'},
            color_discrete_sequence=['#4ECDC4'],
            template='plotly_white'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 标签云（使用Plotly的条形图模拟）
        st.markdown("### 热门标签分析")
        if 'tag1' in df.columns:
            # 过滤掉空标签
            tag_counts = df['tag1'].replace('', pd.NA).dropna().value_counts().head(15)
            if not tag_counts.empty:
                fig = px.bar(
                    x=tag_counts.values,
                    y=tag_counts.index,
                    orientation='h',
                    title='热门标签 Top 15',
                    labels={'x': '出现次数', 'y': '标签'},
                    color=tag_counts.values,
                    color_continuous_scale='Oranges',
                    template='plotly_white'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("没有找到有效的标签数据。")
        else:
            st.warning("数据中缺少 'tag1' 列，无法进行热门标签分析。")

# ---------------------- 主界面布局与逻辑 ----------------------
def main():
    st.title("🎵 网易云歌单数据分析工具")
    st.markdown("---")
    
    df = load_and_preprocess_all_data()
    
    if df.empty:
        st.info("请添加正确格式的CSV文件后重新运行。")
        return
    
    # 显示数据概览
    display_data_overview(df)
    st.markdown("---")
    
    # 筛选条件侧边栏
    with st.sidebar:
        st.header("🔍 筛选条件")
        
        selected_cats = st.multiselect("歌单分类", options=df['分类'].unique(), default=df['分类'].unique())
        
        play_min, play_max = st.slider("播放次数范围",
            min_value=int(df['播放次数'].min()),
            max_value=int(df['播放次数'].max()),
            value=(int(df['播放次数'].min()), int(df['播放次数'].max()))
        )
        
        fav_min = st.number_input("最小收藏量", min_value=0, max_value=int(df['收藏量'].max()), value=0)
        
        # 日期筛选优化
        has_dates = not df['创建日期'].isna().all()
        if has_dates:
            date_min, date_max = st.date_input("创建日期范围",
                value=(df['创建日期'].min(), df['创建日期'].max()),
                min_value=df['创建日期'].min(),
                max_value=df['创建日期'].max()
            )
        else:
            st.warning("数据中缺少有效日期信息")
        
        len_min, len_max = st.slider("歌单歌曲数量",
            min_value=1,
            max_value=int(df['歌单长度'].max()),
            value=(1, int(df['歌单长度'].max()))
        )
    
    # 应用筛选逻辑
    filtered_df = df[
        (df['分类'].isin(selected_cats)) &
        (df['播放次数'] >= play_min) &
        (df['播放次数'] <= play_max) &
        (df['收藏量'] >= fav_min) &
        (df['歌单长度'] >= len_min) &
        (df['歌单长度'] <= len_max)
    ].copy()
    
    if has_dates:
        date_min_ts = pd.to_datetime(date_min)
        date_max_ts = pd.to_datetime(date_max)
        filtered_df = filtered_df[
            (filtered_df['创建日期'] >= date_min_ts) &
            (filtered_df['创建日期'] <= date_max_ts)
        ]
    
    # 结果显示
    st.subheader("📋 筛选结果")
    st.markdown(f"**符合条件的歌单数量：{len(filtered_df)}**")
    
    # 使用卡片样式显示数据表格
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(
        filtered_df[['名称', '分类', '创建日期', '播放次数', '收藏量', '评论数', '歌单长度', 'tag1']],
        height=400,
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 高级可视化
    if not filtered_df.empty:
        plot_advanced_visualizations(filtered_df)
    else:
        st.warning("当前筛选条件下没有找到匹配的数据，无法生成可视化图表。")
    
    # 导出功能
    st.markdown("---")
    st.subheader("💾 结果导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("导出为CSV文件"):
            if not filtered_df.empty:
                export_path = DATA_DIR / "筛选后的歌单数据.csv"
                filtered_df.to_csv(export_path, index=False, encoding='utf-8-sig')
                st.success(f"✅ CSV文件已导出至: {export_path}")
            else:
                st.warning("❌ 没有可导出的数据。")
    
    with col2:
        if st.button("导出为Excel文件"):
            if not filtered_df.empty:
                export_path = DATA_DIR / "筛选后的歌单数据.xlsx"
                filtered_df.to_excel(export_path, index=False, engine='openpyxl')
                st.success(f"✅ Excel文件已导出至: {export_path}")
            else:
                st.warning("❌ 没有可导出的数据。")

# ---------------------- 运行入口 ----------------------
if __name__ == "__main__":
    main()