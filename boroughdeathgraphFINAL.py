import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import sys


def clear_screen():
    """Clear console screen"""
    print("\n" * 50)


def print_header():
    """Print application header"""
    print("=" * 80)
    print("NYC DRUG OVERDOSE DEATHS ANALYSIS (2018-2022)".center(80))
    print("Interactive Visualization System".center(80))
    print("=" * 80)


def load_and_prepare_data():
    """Load and prepare datasets with cleaning and merging"""
    print("\n📂 Loading datasets...")

    # Load data
    income_path = Path(
        "/Users/abdelmahmoud/Downloads/nyc_zipcode_income_acs_2018-2022.csv")
    overdose_path = Path(
        "/Users/abdelmahmoud/Downloads/NYC_Overdose_Deaths_2018_2023.csv")

    df_income = pd.read_csv(income_path)
    df_od = pd.read_csv(overdose_path)

    # Data cleaning
    df_income["Borough"] = df_income["Borough"].astype(
        str).str.strip().str.title()
    df_od["Borough"] = df_od["Borough"].astype(str).str.strip().str.title()

    # Get borough income statistics
    df_income_stats = (
        df_income.groupby("Borough")["Median_Household_Income"]
        .agg(['median', 'mean', 'min', 'max', 'std'])
        .reset_index()
        .rename(columns={
            'median': 'median_income',
            'mean': 'mean_income',
            'min': 'min_income',
            'max': 'max_income',
            'std': 'std_income'
        })
    )

    # Calculate poverty statistics
    df_poverty = (
        df_income.groupby("Borough")["Poverty_Rate"]
        .mean()
        .reset_index()
        .rename(columns={"Poverty_Rate": "avg_poverty_rate"})
    )

    df_income_stats = pd.merge(df_income_stats, df_poverty, on="Borough")

    # Filter to 2018-2022 for income comparison
    df_od_filtered = df_od[df_od["Year"].between(2018, 2022)].copy()

    # Merge datasets
    df = pd.merge(df_od_filtered, df_income_stats, on="Borough", how="inner")

    # Calculate key metrics
    df['period'] = df['Year'].apply(
        lambda x: 'Pre-Pandemic' if x < 2020 else 'Pandemic')

    # Year-over-year change
    df_yoy = df.sort_values(['Borough', 'Year'])
    df_yoy['pct_change'] = df_yoy.groupby(
        'Borough')['Overdose_Deaths'].pct_change() * 100

    # Total statistics by borough
    borough_totals = df.groupby('Borough').agg({
        'Overdose_Deaths': ['sum', 'mean'],
        'Rate_per_100k': ['mean'],
        'median_income': 'first',
        'avg_poverty_rate': 'first'
    }).round(2)
    borough_totals.columns = ['Total_Deaths', 'Avg_Deaths_Year',
                              'Avg_Rate_100k', 'Median_Income', 'Avg_Poverty_Rate']

    print("✅ Data loaded and merged successfully!")
    print(f"   - {len(df)} records processed")
    print(f"   - {len(df['Borough'].unique())} boroughs analyzed")
    print(f"   - Years: {df['Year'].min()} to {df['Year'].max()}")

    return df, df_income_stats, borough_totals


def get_borough_colors():
    """Return consistent color scheme for boroughs"""
    return {
        "Bronx": "#e74c3c",
        "Brooklyn": "#3498db",
        "Manhattan": "#2ecc71",
        "Queens": "#f39c12",
        "Staten Island": "#9b59b6"
    }


def create_hover_annotation(fig, ax, df, borough_colors):
    """Create interactive hover annotations for scatter/line plots"""
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="black", alpha=0.8),
                        arrowprops=dict(arrowstyle="->"),
                        color="white", fontsize=10, fontweight="bold")
    annot.set_visible(False)

    def hover(event):
        if event.inaxes == ax:
            for line in ax.get_lines():
                if line.contains(event)[0]:
                    # Get the borough name from the line label
                    borough = line.get_label()
                    if borough and borough in df['Borough'].values:
                        # Find the closest point
                        xdata, ydata = line.get_data()
                        distances = np.sqrt(
                            (xdata - event.xdata)**2 + (ydata - event.ydata)**2)
                        idx = np.argmin(distances)

                        year = int(xdata[idx])
                        rate = ydata[idx]

                        # Get additional info
                        borough_data = df[(df['Borough'] == borough) & (
                            df['Year'] == year)].iloc[0]
                        income = borough_data['median_income']
                        total_deaths = borough_data['Overdose_Deaths']

                        # Update annotation
                        annot.xy = (xdata[idx], ydata[idx])
                        text = f"{borough}\nYear: {year}\nRate: {rate:.1f}/100k\nDeaths: {total_deaths}\nIncome: ${income:,.0f}"
                        annot.set_text(text)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return

            # Check scatter plots
            for collection in ax.collections:
                cont, ind = collection.contains(event)
                if cont:
                    # Get offset coordinates
                    offsets = collection.get_offsets()
                    if len(ind['ind']) > 0:
                        idx = ind['ind'][0]
                        x, y = offsets[idx]

                        # Find matching data point
                        matching = df[(df['Year'].round() == round(x)) &
                                      (abs(df['Rate_per_100k'] - y) < 1)]

                        if not matching.empty:
                            row = matching.iloc[0]
                            borough = row['Borough']
                            year = int(row['Year'])
                            rate = row['Rate_per_100k']
                            income = row['median_income']
                            total_deaths = row['Overdose_Deaths']

                            annot.xy = (x, y)
                            text = f"{borough}\nYear: {year}\nRate: {rate:.1f}/100k\nDeaths: {total_deaths}\nIncome: ${income:,.0f}"
                            annot.set_text(text)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                            return

            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def create_scatter_hover(fig, ax, borough_totals, borough_colors, mode='income'):
    """Create interactive hover for scatter plots (income or poverty correlation)"""
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="black", alpha=0.8),
                        arrowprops=dict(arrowstyle="->"),
                        color="white", fontsize=10, fontweight="bold")
    annot.set_visible(False)

    # Store borough positions for each collection
    borough_list = list(borough_totals.index)

    def hover(event):
        if event.inaxes == ax:
            vis = annot.get_visible()
            for i, collection in enumerate(ax.collections):
                # Skip the trend line (it's a LineCollection or line)
                if hasattr(collection, 'get_offsets'):
                    cont, ind = collection.contains(event)
                    if cont and len(ind['ind']) > 0:
                        # For scatter plots where each borough is its own collection
                        # The collection index corresponds to the borough
                        if i < len(borough_list):
                            borough = borough_list[i]

                            if mode == 'income':
                                x_val = borough_totals.loc[borough,
                                                           'Median_Income']
                                y_val = borough_totals.loc[borough,
                                                           'Total_Deaths']
                                text = f"{borough}\nIncome: ${x_val:,.0f}\nTotal Deaths: {y_val:,.0f}\n(2018-2022)"
                            else:  # poverty mode
                                x_val = borough_totals.loc[borough,
                                                           'Avg_Poverty_Rate']
                                y_val = borough_totals.loc[borough,
                                                           'Avg_Rate_100k']
                                text = f"{borough}\nPoverty: {x_val:.1f}%\nAvg Rate: {y_val:.1f}/100k"

                            idx = ind['ind'][0]
                            annot.xy = collection.get_offsets()[idx]
                            annot.set_text(text)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                            return

            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def show_all_visualizations(df, df_income_stats, borough_totals):
    """Display all visualizations with interactive hover - POVERTY CHART REMOVED"""
    print("\n📊 Generating comprehensive dashboard with interactive hover...")

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    borough_colors = get_borough_colors()

    # Calculate correlations
    correlation = stats.pearsonr(
        borough_totals['Median_Income'], borough_totals['Total_Deaths'])

    # Graph 1: Rate per 100k over time
    ax1 = fig.add_subplot(gs[0, :2])
    for borough in sorted(df["Borough"].unique()):
        borough_data = df[df["Borough"] == borough].sort_values('Year')
        ax1.plot(borough_data["Year"], borough_data["Rate_per_100k"],
                 marker='o', linewidth=2.5, markersize=8,
                 color=borough_colors.get(borough, "#95a5a6"), label=borough,
                 picker=5)

    ax1.axvspan(2020, 2022, alpha=0.1, color='red', label='Pandemic Period')
    ax1.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Overdose Death Rate per 100k",
                   fontsize=12, fontweight="bold")
    ax1.set_title("NYC Overdose Death Rate Trends (2018-2022) - \nNormalized per 100,000 Population",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_xticks(sorted(df["Year"].unique()))

    # Add interactive hover
    create_hover_annotation(fig, ax1, df, borough_colors)

    # Graph 2: Income vs Total Deaths Correlation
    ax2 = fig.add_subplot(gs[0, 2])
    boroughs = borough_totals.index
    x = borough_totals['Median_Income']
    y = borough_totals['Total_Deaths']

    for borough in boroughs:
        ax2.scatter(borough_totals.loc[borough, 'Median_Income'],
                    borough_totals.loc[borough, 'Total_Deaths'],
                    s=300, color=borough_colors.get(borough, "#95a5a6"),
                    alpha=0.7, edgecolors='black', linewidth=2, label=borough,
                    picker=5)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax2.plot(x, p(x), "r--", alpha=0.5, linewidth=2,
             label=f'Trend (r={correlation[0]:.3f})')
    ax2.set_xlabel("Median Household Income ($)",
                   fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Deaths (2018-2022)", fontsize=11, fontweight="bold")
    ax2.set_title("Income vs Overdose Deaths",
                  fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best', framealpha=0.9, markerscale=0.6)
    ax2.ticklabel_format(style='plain', axis='x')

    # Add interactive hover
    create_scatter_hover(fig, ax2, borough_totals,
                         borough_colors, mode='income')

    # Graph 3: 2018 vs 2022 Comparison
    ax3 = fig.add_subplot(gs[1, :2])
    comparison_data = df[df['Year'].isin([2018, 2022])].pivot(
        index='Borough', columns='Year', values='Rate_per_100k')
    x_pos = np.arange(len(comparison_data.index))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, comparison_data[2018], width,
                    label='2018', alpha=0.8, color='#3498db')
    bars2 = ax3.bar(x_pos + width/2, comparison_data[2022], width,
                    label='2022', alpha=0.8, color='#e74c3c')

    for i, borough in enumerate(comparison_data.index):
        pct_change = ((comparison_data.loc[borough, 2022] - comparison_data.loc[borough, 2018])
                      / comparison_data.loc[borough, 2018] * 100)
        ax3.text(i, comparison_data.loc[borough, 2022] + 2,
                 f'+{pct_change:.0f}%', ha='center', fontsize=9, fontweight='bold')

    ax3.set_xlabel("Borough", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Death Rate per 100k", fontsize=12, fontweight="bold")
    ax3.set_title("Pre-Pandemic (2018) vs Pandemic Peak (2022)\nPercentage Increase by Borough",
                  fontsize=14, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(comparison_data.index, rotation=15, ha='right')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    # Graph 4: Bubble Chart with legend
    ax4 = fig.add_subplot(gs[1, 2])
    for borough in sorted(df["Borough"].unique()):
        borough_data = df[df["Borough"] == borough]
        ax4.scatter(borough_data["Year"], borough_data["Rate_per_100k"],
                    s=borough_data["median_income"] / 150, alpha=0.6,
                    color=borough_colors.get(borough, "#95a5a6"),
                    edgecolors="black", linewidth=0.5, label=borough,
                    picker=5)

    ax4.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Rate per 100k", fontsize=11, fontweight="bold")
    ax4.set_title("Bubble Size = Income\n(2018-2022)",
                  fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle="--")
    ax4.set_xticks(sorted(df["Year"].unique()))
    ax4.legend(fontsize=8, loc='best', framealpha=0.9, markerscale=0.6)

    # Add interactive hover
    create_hover_annotation(fig, ax4, df, borough_colors)

    fig.suptitle('NYC Drug Overdose Deaths Analysis (2018-2022)\nInteractive Dashboard - Hover Over Points for Details',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('nyc_overdose_comprehensive_analysis.png',
                dpi=300, bbox_inches='tight')
    print("✅ Dashboard saved as 'nyc_overdose_comprehensive_analysis.png'")
    print("💡 Hover over any data point to see detailed information!")
    plt.show()


def show_trend_analysis(df):
    """Display death rate trends over time with interactive hover"""
    print("\n📈 Generating trend analysis with interactive hover...")

    fig, ax = plt.subplots(figsize=(14, 8))
    borough_colors = get_borough_colors()

    for borough in sorted(df["Borough"].unique()):
        borough_data = df[df["Borough"] == borough].sort_values('Year')
        ax.plot(borough_data["Year"], borough_data["Rate_per_100k"],
                marker='o', linewidth=3, markersize=10,
                color=borough_colors.get(borough, "#95a5a6"), label=borough,
                picker=5)

    ax.axvspan(2020, 2022, alpha=0.15, color='red', label='Pandemic Period')
    ax.set_xlabel("Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Overdose Death Rate per 100k",
                  fontsize=14, fontweight="bold")
    ax.set_title("NYC Overdose Death Rate Trends by Borough (2018-2022)\nHOVER OVER POINTS FOR DETAILS",
                 fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    ax.set_xticks(sorted(df["Year"].unique()))

    # Add interactive hover
    create_hover_annotation(fig, ax, df, borough_colors)

    plt.tight_layout()
    plt.savefig('trend_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Chart saved as 'trend_analysis.png'")
    print("💡 Hover over any data point to see detailed information!")
    plt.show()


def show_correlation_analysis(borough_totals):
    """Display income correlation with deaths - POVERTY CHART REMOVED"""
    print("\n🔍 Generating correlation analysis with interactive hover...")

    fig, ax = plt.subplots(figsize=(10, 8))
    borough_colors = get_borough_colors()
    boroughs = borough_totals.index

    # Income correlation only
    x_income = borough_totals['Median_Income']
    y_deaths = borough_totals['Total_Deaths']
    correlation = stats.pearsonr(x_income, y_deaths)

    for borough in boroughs:
        ax.scatter(borough_totals.loc[borough, 'Median_Income'],
                   borough_totals.loc[borough, 'Total_Deaths'],
                   s=500, color=borough_colors.get(borough, "#95a5a6"),
                   alpha=0.7, edgecolors='black', linewidth=2, label=borough,
                   picker=5)

    z = np.polyfit(x_income, y_deaths, 1)
    p = np.poly1d(z)
    ax.plot(x_income, p(x_income), "r--", alpha=0.6,
            linewidth=3, label='Trend Line')
    ax.set_xlabel("Median Household Income ($)",
                  fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Deaths (2018-2022)", fontsize=14, fontweight="bold")
    ax.set_title(f"Income vs Deaths - HOVER FOR DETAILS\nPearson r = {correlation[0]:.3f} (p={correlation[1]:.4f})",
                 fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='x')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)

    # Add interactive hover
    create_scatter_hover(fig, ax, borough_totals,
                         borough_colors, mode='income')

    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Chart saved as 'correlation_analysis.png'")
    print("💡 Hover over any data point to see detailed information!")
    plt.show()


def show_comparison_chart(df):
    """Display 2018 vs 2022 comparison"""
    print("\n📊 Generating comparison chart...")

    fig, ax = plt.subplots(figsize=(12, 8))
    comparison_data = df[df['Year'].isin([2018, 2022])].pivot(
        index='Borough', columns='Year', values='Rate_per_100k')
    x_pos = np.arange(len(comparison_data.index))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, comparison_data[2018], width,
                   label='2018 (Pre-Pandemic)', alpha=0.9, color='#3498db', edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, comparison_data[2022], width,
                   label='2022 (Pandemic Peak)', alpha=0.9, color='#e74c3c', edgecolor='black')

    for i, borough in enumerate(comparison_data.index):
        pct_change = ((comparison_data.loc[borough, 2022] - comparison_data.loc[borough, 2018])
                      / comparison_data.loc[borough, 2018] * 100)
        ax.text(i, comparison_data.loc[borough, 2022] + 3,
                f'+{pct_change:.0f}%', ha='center', fontsize=11, fontweight='bold', color='red')

    ax.set_xlabel("Borough", fontsize=14, fontweight="bold")
    ax.set_ylabel("Death Rate per 100,000 Population",
                  fontsize=14, fontweight="bold")
    ax.set_title("Overdose Death Rate: 2018 vs 2022 Comparison\nPercentage Increase by Borough",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_data.index, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('comparison_2018_2022.png', dpi=300, bbox_inches='tight')
    print("✅ Chart saved as 'comparison_2018_2022.png'")
    plt.show()


def show_statistics(df, df_income_stats, borough_totals):
    """Display comprehensive statistics"""
    correlation = stats.pearsonr(
        borough_totals['Median_Income'], borough_totals['Total_Deaths'])
    corr_pov = stats.pearsonr(
        borough_totals['Avg_Poverty_Rate'], borough_totals['Avg_Rate_100k'])
    comparison_data = df[df['Year'].isin([2018, 2022])].pivot(
        index='Borough', columns='Year', values='Rate_per_100k')

    print("\n" + "="*80)
    print("COMPREHENSIVE BOROUGH STATISTICS (2018-2022)")
    print("="*80)
    print(borough_totals.to_string())

    print("\n" + "="*80)
    print("INCOME DISTRIBUTION BY BOROUGH")
    print("="*80)
    income_display = df_income_stats[['Borough', 'min_income', 'median_income',
                                      'max_income', 'std_income', 'avg_poverty_rate']]
    income_display.columns = ['Borough', 'Min Income', 'Median Income',
                              'Max Income', 'Std Dev', 'Avg Poverty %']
    print(income_display.to_string(index=False))

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(
        f"• Income vs Deaths correlation: r = {correlation[0]:.3f} (p = {correlation[1]:.4f})")
    print(
        f"• Poverty vs Death Rate correlation: r = {corr_pov[0]:.3f} (p = {corr_pov[1]:.4f})")
    print(
        f"• Highest total deaths: {borough_totals['Total_Deaths'].idxmax()} ({borough_totals['Total_Deaths'].max():.0f})")
    print(
        f"• Highest avg rate per 100k: {borough_totals['Avg_Rate_100k'].idxmax()} ({borough_totals['Avg_Rate_100k'].max():.1f})")
    print(
        f"• Lowest median income: {borough_totals['Median_Income'].idxmin()} (${borough_totals['Median_Income'].min():,.0f})")
    print(
        f"• Average increase 2018→2022: {comparison_data[2022].mean() - comparison_data[2018].mean():.1f} deaths per 100k")

    pct_increases = (
        (comparison_data[2022] - comparison_data[2018]) / comparison_data[2018] * 100)
    largest_increase_borough = pct_increases.idxmax()
    largest_increase_value = pct_increases.max()
    print(
        f"• Largest percentage increase: {largest_increase_borough} (+{largest_increase_value:.1f}%)")


def main_menu():
    """Display main menu and handle user choices"""
    # Load data once
    df, df_income_stats, borough_totals = load_and_prepare_data()

    while True:
        clear_screen()
        print_header()
        print("\n📋 MENU OPTIONS:")
        print("  1. View All Visualizations")
        print("  2. View Trend Analysis")
        print("  3. View Income Correlation Analysis")
        print("  4. View 2018 vs 2022 Comparison")
        print("  5. View Statistical Summary")
        print("  6. Export All Charts")
        print("  0. Exit")
        print("\n" + "="*80)
        print("💡 TIP: Hover over data points in charts to see detailed information!")
        print("="*80)

        choice = input("\nEnter your choice (0-6): ").strip()

        if choice == '1':
            show_all_visualizations(df, df_income_stats, borough_totals)
        elif choice == '2':
            show_trend_analysis(df)
        elif choice == '3':
            show_correlation_analysis(borough_totals)
        elif choice == '4':
            show_comparison_chart(df)
        elif choice == '5':
            show_statistics(df, df_income_stats, borough_totals)
        elif choice == '6':
            print("\n📦 Exporting all charts...")
            show_all_visualizations(df, df_income_stats, borough_totals)
            show_trend_analysis(df)
            show_correlation_analysis(borough_totals)
            show_comparison_chart(df)
            print("\n✅ All charts exported successfully!")
        elif choice == '0':
            print("\n👋 Thank you for using the NYC Overdose Analysis System!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please enter a number between 0 and 6.")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main_menu()
