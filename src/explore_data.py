import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

def draw_box_plot(df, col, title):
    plt.figure(figsize=(10, 6))
    upper_limit = df[col].quantile(0.90)
    sns.boxplot(x=df[df[col] <= upper_limit][col])
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.title(title)
    plt.show()

def print_title(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def print_sub_title(title):
    print("*" * 25 + title + "*" * 25 + "\n")

def explore_data(df):
    print_title("Exploring data")
    print_sub_title("Data set head")
    print(df.head())
    df_as_list = df.columns.tolist()
    sorted_df_as_list = sorted(df_as_list)
    print_sub_title("Data set columns")
    print(sorted_df_as_list)
    draw_box_plot(df, 'salary_usd', 'Salary USD - salary_usd - bottom 90%')
    draw_box_plot(df, 'years_experience', 'Years Experience - years_experience - bottom 90%')
    draw_box_plot(df, 'benefits_score', 'Benefits Score - benefits_score - bottom 90%')

    plt.figure(figsize=(10, 6))
    category_counts = df['employment_type'].value_counts()  # sorted for easier understanding
    sns.barplot(y=category_counts.index, x=category_counts.values,
                order=category_counts.index)  # tried countplot but it was kind of messy.
    plt.title('Employment Type - employment_type')
    plt.show()

    plt.figure(figsize=(10, 6))
    category_counts = df['job_title'].value_counts()  # sorted for easier understanding
    sns.barplot(y=category_counts.index, x=category_counts.values, order=category_counts.index)
    plt.title('Job Title - job_title')
    plt.show()

