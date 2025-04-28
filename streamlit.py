import streamlit as st
import pandas as pd
import mysql
from mysql import connector
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


#MySQL connector
con = connector.connect(
    host="localhost",
    user="root",
    password="Saisenthur@13"
    )

mycursor=con.cursor()

mycursor.execute('USE PhonePe')


#STREAMLIT APPLICATION CODE :

def home():
    st.title("PhonePe Transaction")
    st.title("Insights 	:heavy_dollar_sign: :money_with_wings: :chart:")

    mycursor.execute("select State, sum(amount) as Total_Amount from agg_transaction group by state order by Total_Amount desc")
    data=mycursor.fetchall()
    columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
    df1 = pd.DataFrame(data, columns=columns)

    fig = px.choropleth(
    df1,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='State',
    color='Total_Amount',
    color_continuous_scale='purples'
    )

    fig.update_geos(fitbounds="locations", visible=False)
    #fig.update_layout(height=600, width=1250)

    # Show the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.table(df1)


def Data_Analysis():
    st.title("Data Analytics and Visualization :")
    
    def cs1(): #cs1)Decoding Transaction Dynamics on PhonePe

        #1)
        st.write('### 1) State wise Transactions')

        mycursor.execute("select State, Sum(count) as Total_Count, sum(amount) as Total_Amount from agg_transaction group by state order by Total_Amount desc")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        plt.figure(figsize=(10, 5)) 
        sns.barplot(x='State', y='Total_Amount', data=df1, )

        # Add title and labels
        plt.title('State wise Total Transaction Amount', fontsize = 13, fontweight='bold')
        plt.xlabel('State', fontsize = 10, fontweight='bold')
        plt.ylabel('Total_Amount', fontsize = 10, fontweight='bold')
        plt.xticks(rotation=90)

        # Display the plot
        st.pyplot(plt)

        plt.clf()

        #2)
        st.write("### 2) Yearly Transaction Growth")

        mycursor.execute("SELECT year, SUM(amount) AS total_amount FROM agg_transaction GROUP BY year ORDER BY year")
        data = mycursor.fetchall()
        df1 = pd.DataFrame(data, columns=["year", "total_amount"])

        plt.plot(df1['year'], df1['total_amount'], marker='o', color='green')
        plt.title(f"Yearly Transaction Value", fontsize = 14, fontweight='bold')
        plt.xlabel("Year", fontsize = 12)
        plt.ylabel("Total Amount (INR)", fontsize = 12)
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()

        #3)
        st.write('### 3) Transaction Type')

        mycursor.execute("select Transaction_type, Sum(count) as Total_Count, sum(amount) as Total_Amount from agg_transaction group by Transaction_type order by Total_Amount desc")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        plt.figure(figsize=(10, 10))  # Larger figure for more space
        explode = [0.05] * len(df1)  # Slightly explode all slices for spacing
        plt.pie(df1['Total_Amount'], explode=explode, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})

        plt.title("Transaction Amount by Type", fontsize = 15, fontweight='bold')
        plt.legend(df1['Transaction_type'], loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

        #4)
        st.write('### 4) Quarterly Transaction Amounts')

        mycursor.execute("SELECT DISTINCT Year FROM agg_transaction ORDER BY Year")
        years = [row[0] for row in mycursor.fetchall()]

        mycursor.execute("SELECT DISTINCT State FROM agg_transaction")
        states = [row[0] for row in mycursor.fetchall()]

        selected_year = st.selectbox("Select Year", years)
        selected_state = st.selectbox("Select State", sorted(states))

        query = """
        SELECT Quarter, SUM(amount) AS Total_Amount
        FROM agg_transaction
        WHERE Year = %s AND State = %s
        GROUP BY Quarter
        ORDER BY Quarter
        """
        mycursor.execute(query, (selected_year, selected_state))
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result

        # Plot Line Graph
        if data:
            df1 = pd.DataFrame(data, columns=columns)
            
            plt.figure(figsize=(12, 6))
            plt.plot(df1["Quarter"], df1["Total_Amount"], marker='o')
            plt.title(f"Quarter-wise Total Transaction Amount in {selected_state} {selected_year}", fontsize = 14, fontweight='bold')
            plt.xlabel("Quarter", fontsize = 12)
            plt.ylabel("Transaction Amount", fontsize = 12)
            plt.grid(True)
            st.pyplot(plt)
            plt.clf()
        else:
            st.write("No data available for the selected year and state.")
        
        

    def cs2(): #cs2)Device Dominance and User Engagement Analysis

        #1)
        st.write('### 1) Brand wise Registered Users')

        mycursor.execute("select Brand, Year, Sum(Count) as Total_RegisteredUsers from agg_user group by Brand,Year order by Year ")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        fig = px.bar(
        df1,
        x="Brand",
        y="Total_RegisteredUsers",
        color="Year",
        title="Brand-wise Registered Users (Stacked by Year)",
        labels={"Total_RegisteredUsers": "Registered Users"},
        barmode="stack",
        height=650,
        )

        fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels for clarity

        st.plotly_chart(fig, use_container_width=True)
        plt.clf()

        #2)
        st.write('### 2) Total AppOpens Users by Device Type')

        mycursor.execute("select au.Brand as Brand, Sum(mu.AppOpens) as Total_AppOpens from agg_user au left join map_user mu on au.state = mu.state group by au.Brand order by Total_AppOpens desc")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        # Calculate percentages
        total = df1['Total_AppOpens'].sum()
        df1['Percentage'] = (df1['Total_AppOpens'] / total) * 100
        df1['Label'] = df1.apply(lambda row: f"{row['Brand']} - {row['Percentage']:.1f}%", axis=1)

        plt.figure(figsize=(10, 10))  # Larger figure for more space
        explode = [0.05] * len(df1)  # Slightly explode all slices for spacing
        plt.pie(df1['Total_AppOpens'], explode=explode, startangle=90, textprops={'fontsize': 10})

        plt.title("Registered Users by Device Type", fontsize=12)
        plt.legend(df1['Label'], loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

        #3) 
        st.write("### 3) Popular Device Brand by State")
        mycursor.execute("SELECT state, brand, SUM(count) AS total_users FROM agg_user GROUP BY state, brand ORDER BY state, total_users DESC;")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        # Get the brand with max users per state
        top_brands_per_state = df1.loc[df1.groupby("state")["total_users"].idxmax()].reset_index(drop=True)

        brand_summary = top_brands_per_state.groupby("brand").agg(
            state_count=("brand", "count"),               # Count of states
            total_users=("total_users", "sum")            # Total users across those states
        )

        plt.figure(figsize=(10, 5)) 
        sns.barplot(x='brand', y='state_count', data=brand_summary, )
        # Add title and labels
        plt.title('Popular Device')
        plt.xlabel('State')
        plt.ylabel('Total_Amount')
        plt.xticks(rotation=90)

        # Display the plot
        st.pyplot(plt)
        plt.clf()

        st.table(top_brands_per_state)

        #4)
        st.write("### 4) Engagement Efficiency")
        query = """
        select au.brand as Brand, au.Year as Year, Sum(mu.RegisteredUsers) as Total_RegisteredUsers, Sum(mu.appOpens) as Total_AppOpens, ROUND(SUM(mu.appOpens) / SUM(mu.registeredUsers), 2) AS Opens_Per_User 
        from agg_user au
        left join map_user mu on au.state = mu.state
        group by Brand,Year 
        order by Brand,Year desc, opens_per_user desc"""
        mycursor.execute(query)
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        # Brand selector
        Brands = sorted(df1['Brand'].unique())
        Brands.insert(0, "All Brands")
        selected_brand = st.selectbox("Select a Brand or All Brands :", Brands)

        # Plotting
        plt.figure(figsize=(12, 6))

        if selected_brand == "All Brands":
            pivot_df = df1.pivot(index='Year', columns='Brand', values='Opens_Per_User')
            for brand in pivot_df.columns:
                plt.plot(pivot_df.index, pivot_df[brand], label=brand, marker='o')
            plt.title("Engagement Efficiency: Opens Per User Over Years (All Brands)")
        else:
            filtered_df = df1[df1['Brand'] == selected_brand]
            plt.plot(filtered_df['Year'], filtered_df['Opens_Per_User'], marker='o', label=selected_brand)
            plt.title(f"Engagement Efficiency: Opens Per User Over Years ({selected_brand})")

        plt.xlabel("Year")
        plt.ylabel("Opens Per User")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

        st.table(df1[['Brand','Year', 'Opens_Per_User']])

    def cs3() : #3) Insurance Penetration and Growth Potential Analysis

        #1)
        st.write('### 1) State-wise Insurance Transaction Volume & Value')

        mycursor.execute("SELECT state, SUM(count) AS total_policies, SUM(amount) AS total_premium FROM agg_insurance GROUP BY state ORDER BY total_policies DESC limit 5")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        col1, col2 = st.columns(2)

        with col1:
            plt.figure(figsize=(4.75, 5)) 
            sns.barplot(x='state', y='total_policies', data=df1, color='red')

            # Add title and labels
            plt.title('State wise Total Policies', fontsize = 13, fontweight='bold')
            plt.xlabel('state', fontsize = 10, fontweight='bold')
            plt.ylabel('total_policies', fontsize = 10, fontweight='bold')
            plt.xticks(rotation=90)

            # Display the plot
            st.pyplot(plt)

            plt.clf()

        with col2:
            plt.figure(figsize=(5, 5)) 
            sns.barplot(x='state', y='total_premium', data=df1, color='orange' )

            # Add title and labels
            plt.title('State wise Total Premium Amount', fontsize = 13, fontweight='bold')
            plt.xlabel('state', fontsize = 10, fontweight='bold')
            plt.ylabel('total_premium', fontsize = 10, fontweight='bold')
            plt.xticks(rotation=90)

            # Display the plot
            st.pyplot(plt)

            plt.clf()

        #2)
        st.write('### 2) Quarterly Growth in Insurance Transactions') #most insurance are taken during 4th quarter

        mycursor.execute("SELECT DISTINCT year FROM agg_insurance ORDER BY year")
        years = [row[0] for row in mycursor.fetchall()]
        selected_year = st.selectbox("Select Year", years)

        query = """
            SELECT quarter, SUM(count) AS total_policies
            FROM agg_insurance
            WHERE year = %s
            GROUP BY quarter
            ORDER BY quarter
        """
        mycursor.execute(query, (selected_year,))
        data = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        df1 = pd.DataFrame(data, columns=columns)

        # Plot
        plt.figure(figsize=(8, 8))
        explode = [0.05] * len(df1)
        plt.pie(df1['total_policies'], explode=explode, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        plt.title(f"Quarterly Insurance Transactions in {selected_year}", fontsize=15, fontweight='bold')
        plt.legend(df1['quarter'], loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()


        #3)
        st.write('### 3) Top Performing States by Insurance Value per Policy')

        mycursor.execute("SELECT state, ROUND(SUM(amount)/SUM(count), 2) AS avg_policy_value FROM agg_insurance GROUP BY state ORDER BY avg_policy_value DESC")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        #st.table(df1)

        plt.figure(figsize=(10, 12))
        plt.barh(df1['state'], df1['avg_policy_value'], color='teal')
        plt.xlabel("Average Policy Value (INR)")
        plt.ylabel("State")
        plt.title("Average Insurance Policy Value by State", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest value at the top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        st.pyplot(plt)
        plt.clf()

        #4) 
        st.write('### 4) Insurance vs Total Transactions Ratio') #check the table for more insights into higher transactions and lower insurance counts

        mycursor.execute("SELECT ai.state, SUM(ai.amount) AS insurance_amount, SUM(at.amount) AS total_transaction_amount, ROUND(SUM(ai.amount) / SUM(at.amount) * 100, 2) AS insurance_penetration_percentage FROM agg_insurance ai JOIN agg_transaction at ON ai.state = at.state AND ai.year = at.year AND ai.quarter = at.quarter GROUP BY ai.state ORDER BY insurance_penetration_percentage DESC")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        st.table(df1[['state', 'insurance_penetration_percentage']])


    def cs4(): #cs4) Transaction Analysis for Market Expansion
        
        #1)
        st.write('### 1) States wise Transaction Year-over-Year')

        mycursor.execute("SELECT DISTINCT state FROM agg_transaction ORDER BY state")
        states = [row[0] for row in mycursor.fetchall()]
        states.insert(0, "All States")

        selected_state = st.selectbox("Select State", states)

        if selected_state == "All States":
            mycursor.execute("""
                SELECT state, year, SUM(amount) AS total_amount
                FROM agg_transaction
                GROUP BY state, year
                ORDER BY state, year;
            """)
        else:
            mycursor.execute("""
                SELECT state, year, SUM(amount) AS total_amount
                FROM agg_transaction
                WHERE state = %s
                GROUP BY state, year
                ORDER BY year;
            """, (selected_state,))

        # Fetch data
        data = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        df1 = pd.DataFrame(data, columns=columns)

        # Plot
        plt.figure(figsize=(12, 6))
        if selected_state == "All States":
            sns.lineplot(data=df1, x="year", y="total_amount", hue="state", marker="o")
            #plt.legend(title="State", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
            plt.legend(title="State",loc='upper center',bbox_to_anchor=(0.5, -0.15),ncol=6 ,fontsize='small',title_fontsize='small')
        else:
            sns.lineplot(data=df1, x="year", y="total_amount", marker="o", label=selected_state)
            plt.legend(title="State", loc="best")

        plt.title("Transaction Growth by Year", fontsize=18, fontweight='bold')
        plt.xlabel("Year", fontsize=14)
        plt.ylabel("Total Transaction Amount (in ₹)", fontsize=14)
        st.pyplot(plt)
        plt.clf()

        #2)
        st.write('### 2) States with Consistently Low Transactions')

        mycursor.execute("SELECT state, SUM(amount) AS total_amount FROM agg_transaction GROUP BY state ORDER BY total_amount limit 20")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        plt.figure(figsize=(10, 12))
        plt.barh(df1['state'], df1['total_amount'], color='teal')
        plt.xlabel("Total_amount", fontsize=12)
        plt.ylabel("State", fontsize=12)
        plt.title("States with Consistently Low Transactions", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest value at the top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        st.pyplot(plt)
        plt.clf()
        
        #3)
        st.write('### 3) States with Low Transaction-to-User Ratio')
        query = """
        SELECT 
            at.state,
            SUM(at.amount) AS total_transaction_amount,
            SUM(au.count) AS total_users,
            ROUND(SUM(at.amount) / SUM(au.count), 2) AS transaction_user_ratio
        FROM agg_transaction at
        JOIN agg_user au ON at.state = au.state AND at.year = au.year AND at.quarter = au.quarter
        GROUP BY at.state
        HAVING total_users > 0
        ORDER BY transaction_user_ratio ;
        """
        mycursor.execute(query)
        data = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        df1 = pd.DataFrame(data, columns=columns)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.bar(df1['state'], df1['transaction_user_ratio'], color='limegreen')
        plt.xlabel('Transaction-to-User Ratio')
        plt.ylabel('State')
        plt.title('States wise Transaction-to-User Ratio', fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=1)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        plt.clf()

        st.table(df1[['state','transaction_user_ratio']])


        #4)

        st.write("### 4) Least Popular Transaction Type by State") #they dont trust our payment portal for financial services
        mycursor.execute("SELECT state, transaction_type, SUM(amount) AS total_amount FROM agg_transaction GROUP BY state, transaction_type ORDER BY state, total_amount ;")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)


        # Get the brand with max users per state
        low_trans_type_per_state = df1.loc[df1.groupby("state")["total_amount"].idxmin()].reset_index(drop=True)

        type_summary = low_trans_type_per_state.groupby("transaction_type").agg(
            state_count=("transaction_type", "count"),               # Count of states
            total_users=("total_amount", "sum")            # Total users across those states
        )

        plt.figure(figsize=(10, 5)) 
        sns.barplot(x='transaction_type', y='state_count', data=type_summary, )
        # Add title and labels
        plt.title('Least Popular Transaction Type', fontsize=14, fontweight='bold')
        plt.xlabel('State', fontsize=12)
        plt.ylabel('Total_Amount', fontsize=12)

        # Display the plot
        st.pyplot(plt)
        plt.clf()

        st.table(low_trans_type_per_state)

    def cs5(): #cs5) User Engagement and Growth Strategy

        #1)

        st.write('### 1) State wise Registered Users')

        mycursor.execute("select State, Year, Sum(RegisteredUsers) as Total_RegisteredUsers from map_user group by state,Year order by Year ")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        fig = px.bar(
        df1,
        x="State",
        y="Total_RegisteredUsers",
        color="Year",
        title="State-wise Registered Users (Stacked by Year)",
        labels={"Total_RegisteredUsers": "Registered Users"},
        barmode="stack",
        height=650
        )

        fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels for clarity

        st.plotly_chart(fig, use_container_width=True)
        plt.clf()

        #2)

        st.write('### 2) State wise District with maximum Registered Users')
        mycursor.execute("select State, District, Sum(RegisteredUsers) as Total_RegisteredUsers from top_user group by State, District order by Total_RegisteredUsers desc")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        max_users_per_state = df1.loc[df1.groupby('State')['Total_RegisteredUsers'].idxmax()]
        max_users_per_state = max_users_per_state.sort_values(by='Total_RegisteredUsers', ascending=False)
        max_users_per_state = max_users_per_state.reset_index(drop=True)

        plt.figure(figsize=(10, 12))
        plt.barh(max_users_per_state['District'], max_users_per_state['Total_RegisteredUsers'], color='purple')
        plt.xlabel("Total_RegisteredUsers", fontsize=12)
        plt.ylabel("District", fontsize=12)
        plt.title("District wise Registered users", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest value at the top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        st.pyplot(plt)
        plt.clf()

        st.table(max_users_per_state)

        #3)

        st.write('### 3) Year wise Total Users and AppOpens ')
        mycursor.execute("select Year, Sum(RegisteredUsers) as Total_RegisteredUsers, sum(AppOpens) as Total_AppOpens from map_user group by Year order by Total_RegisteredUsers desc, Total_AppOpens desc")
        data=mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]  # Extract column names from the query result
        df1 = pd.DataFrame(data, columns=columns)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(12,6))

        # Primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Total Registered Users', color=color)
        ax1.plot(df1['Year'], df1['Total_RegisteredUsers'], marker='o', color=color, label='Total Registered Users')
        ax1.tick_params(axis='y', labelcolor=color)

        # Secondary y-axis
        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel('Total App Opens', color=color)
        ax2.plot(df1['Year'], df1['Total_AppOpens'], marker='o', color=color, label='Total App Opens')
        ax2.tick_params(axis='y', labelcolor=color)

        # Title and Layout
        plt.title('Year-wise Total Registered Users and App Opens')
        fig.tight_layout()
        st.pyplot(fig)
        plt.clf()

        st.table(df1)

        #4)

        st.write("### 4) State-wise Engagement Efficiency")

        query = """
        SELECT 
            state AS State, 
            year AS Year, 
            SUM(registeredUsers) AS Total_RegisteredUsers, 
            SUM(appOpens) AS Total_AppOpens,
            ROUND(SUM(appOpens) / SUM(registeredUsers), 2) AS Opens_Per_User
        FROM 
            map_user
        GROUP BY 
            State, Year
        ORDER BY 
            State, Year DESC
        """
        mycursor.execute(query)
        data = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        df1 = pd.DataFrame(data, columns=columns)

        # State selector
        States = sorted(df1['State'].unique())
        States.insert(0, "All States")
        selected_state = st.selectbox("Select a State or All States :", States)

        # Plotting
        plt.figure(figsize=(12, 6))

        if selected_state == "All States":
            pivot_df = df1.pivot(index='Year', columns='State', values='Opens_Per_User')
            for state in pivot_df.columns:
                plt.plot(pivot_df.index, pivot_df[state], label=state, marker='o')
            plt.title("Engagement Efficiency: Opens Per User Over Years (All States)")
        else:
            filtered_df = df1[df1['State'] == selected_state]
            plt.plot(filtered_df['Year'], filtered_df['Opens_Per_User'], marker='o', label=selected_state)
            plt.title(f"Engagement Efficiency: Opens Per User Over Years ({selected_state})")

        plt.xlabel("Year")
        plt.ylabel("Opens Per User")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()










    case_study = {
        "1) Decoding Transaction Dynamics on PhonePe" : cs1,
        "2) Device Dominance and User Engagement Analysis" : cs2,
        "3) Insurance Penetration and Growth Potential Analysis" : cs3,
        "4) Transaction Analysis for Market Expansion " : cs4,
        "5) User Engagement and Growth Strategy" : cs5
    }

    selected_case_study = st.selectbox("Chose a Buisness Case Study", options= case_study.keys())

    case_study[selected_case_study]()


page_names_func = {
    "Home Page" : home,
    "Data Analysis" : Data_Analysis
}

selected_page = st.sidebar.selectbox("Chose a page", options= page_names_func.keys())

page_names_func[selected_page]()
