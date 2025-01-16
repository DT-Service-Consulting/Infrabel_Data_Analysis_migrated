import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
import streamlit as st
import streamlit.components.v1 as components
# Import your preprocessing, analysis, and visualization functions
from preprocess import *
from visualization import *
from analysis import *

# Streamlit app title and description
st.title("Infrabel Data Analysis Report")

# Create two columns: one for the map and one for the color map info
col1, col2 = st.columns([5, 1])  # Adjust the width ratio as needed

# Column 1: Display the HTML map
with col1:
    st.write("## ETCS Map")
    # Read the HTML map file
    with open('Images/belgium_etcs_map (16).html', 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()

    # Display the HTML map using streamlit's components.html
    components.html(html_content, height=600, scrolling=True)

# Column 2: Display the color map information
with col2:
    st.write("### ETCS Level Colors")
    
    # Color mapping information
   
    st.markdown("""
    - **🟦 ETCS L1 FS**: Blue
    - **🟧 ETCS L2 FS**: Orange
    - **🟨 ETCS L1 LS**: Yellow
    - **🟩 ETCS 1+2**: Green
    - **🟦 TVM-430**: Cyan
    """)

st.markdown("""
This analysis gives a report on train delays, incident hotspots, and the **impact of ETCS (European Train Control System)** on the Belgian railway network, focusing on data from **October 1st, 2024**.

### Key Findings:

##### 🕒 **Delay Patterns**:
- Stations near the borders with **Netherlands** and **Germany** faced the longest delays.
- **Line 50A/3** consistently reported high delay times.

##### 🔴 **Incident Hotspots**:
- **"Intrusion in the lines"** was the most frequent cause of delays.
- **Line 50A/3** reported the most incidents, correlating with frequent delays.

##### 🚦 **ETCS Impact**:
- Line 50A/3, with **ETCS L2 FS**, faced regular delays but showed **faster recovery times** compared to non-ETCS lines during incidents.

##### ⚖️ **ETCS vs. Non-ETCS**:
- Over a 5-year period, **ETCS-equipped tracks** recovered faster from incidents, especially in **winter**, compared to non-ETCS lines.

##### 📏 **Track Length & Delays**:
- Track length had **no significant impact** on delay times, suggesting other factors like **incidents** or **infrastructure issues** are more important.

##### 🚉 **Departure Delays**:
- If a train departs late, it is **unlikely to recover**, resulting in **delayed arrivals**.

##### 🤖 **Predictive Modeling**:
- **ETCS** accounts for **27%** of the variance in delay times, with an **R-squared** value of **0.58**, showcasing its predictive significance.

""", unsafe_allow_html=True)
# 1. Delay Patterns by Station and Tracks
st.header("1. Delay Patterns by Station and Tracks")
st.markdown("""
The visualization shows in **red circles** the stations with maximum waiting time, proportional to delay time. The lines with the highest delay timings are marked in **light red**. Magenta circles correspond to **incident hotspots** based on the number of incidents.
Key observations:
- Stations with the highest delays are located near the borders with **Netherlands** and **Germany**.
- Tracks near the borders also exhibit high delay timings.
            
**Caution**: This is based on one day of data (October 1st, 2024).
""")

# 2. Impact of Incidents on Delays
st.header("2. Impact of Incidents on Delays")
st.markdown("""
The incident that causes the **maximum delay** is "Intrusion in the lines." The **highest number of incidents** and delays occur on **Line 50A/3**.
Even if there is no particular incident, Line 50A/3 faces regular delays, possibly due to being a **crossroad of two major lines**.
""")

# Insert figure for incident impact
st.image('Images/Common_incidents.png', caption="Impact of Incidents on Delays")
st.write("The most common incident to happen on the tracks in the **Intrusion into the tracks** followed by **Collision with a person**. Both of these incidents can be prevented using proper guard rails along the tracks.")
st.image('Images/top_10_lines.png', caption="Top 10 lines with Delay")
st.image('Images/heatmap.png', caption="Primary Types of Delay at Key Lines")
st.write("""**Line 50A** is the most incident-prone, with significant delays due to a wide range of causes, especially electrical supply disruptions and intrusion in the tracks.
**Line 36N** experiences frequent delays related to infrastructure works and intrusions, suggesting that maintenance and security improvements are needed.
Weather conditions and freight train derailments emerge as notable factors on other lines, such as **Line 161** and **Line 124**, respectively.

**Actionable Insights**:
         
**Security improvements** on Lines 50A and 36N to reduce track intrusions.
         
**Infrastructure upgrades**, particularly electrical systems on Line 50A.
         
**Better scheduling of maintenance** and infrastructure works, especially on Line 36N, to minimize disruptions.
         
**Prepare for weather-related disruptions** on Line 161, especially during winter months.
         """)
st.image('Images/50A3.png', caption="Location of 50A/3")
st.write("Even if there is no specific incident, **Line 50A/3** faces regular delays, likely due to its position as a crossroad of two major lines.")
# 3. Role of ETCS on Line 50A/3
st.header("3. Role of ETCS on Line 50A/3")
st.markdown("""
After analyzing the **ETCS level** of Line 50A/3, it was found to be at **L2 FS level**, which also corresponds to the **highest number of incidents per line**.
Despite the frequent incidents, ETCS helps in faster recovery, though there is **no significant daily impact** observed due to **small sample size**, as tested by the **Mann-Whitney U test** (U-statistic: 1253.0000
P-value: 0.6032).
""")

# Insert figure for ETCS role on Line 50A/3
st.image('Images/weighted_incidents.png', caption="Incidents weighted per line on ETCS levels")
st.image('Images/recovery.png', caption="Recovery levels on Non-ECTS vs ECTS levels")
st.image('Images/daily_delay.png', caption="Daily Delay per Line")
st.write(" For this particular day’s data, there is no significant difference in the average departure delays between ETCS and non-ETCS lines. Both categories show similar delay patterns, suggesting that the presence of ETCS did not lead to a noticeable improvement in punctuality on this occasion confirmed by Statistical analysis. ")

# 4. ETCS vs. Non-ETCS Lines Over 5 Years
st.header("4. ETCS vs. Non-ETCS Lines Over 5 Years")
st.markdown("""
Over the past 5 years, during incidents, **tracks with ETCS recover faster** compared to non-ETCS lines.
In winter, **non-ETCS lines** experience much longer delays than ETCS lines.
""")

# Insert figure for ETCS vs. Non-ETCS over time
st.image('Images/over 5 years.png', caption="ETCS vs. Non-ETCS Lines Over 5 Years")

# 5. Length of Track vs. Delays
st.header("5. Length of Track vs. Delays")
st.markdown("""
The **length of the track** does not seem to have any significant impact on delay timings.
""")

# Insert figure for track length vs. delay
st.image('Images/correlation.png', caption="Track Length vs. Delay Timings")

# 6. Delays from Departure to Arrival
st.header("6. Delays from Departure to Arrival")
st.markdown("""
If a train is **late at departure**, it generally **does not recover** during its journey, resulting in a **late arrival** as well.
""")
st.write("""
         #### Proposed Solutions:
   - **Real-Time Rescheduling**: Implement real-time adjustments for delayed trains by optimizing passing times and reducing dwell times.
   - **Prioritization of Late Trains**: Prioritize late trains over on-time trains on shared tracks, allowing recovery without cascading delays.
""")
st.write("""
   - **Adding Buffer Time**: Include buffer times between critical points on delay-prone lines to allow recovery of minor delays.
   - **Strategic Buffer Zones**: Designate segments of the journey for slightly faster speeds to help trains recover lost time.
""")
st.write("""
   - **Flexible Speed Limits**: Allow trains to exceed normal speed limits in certain sections to recover delays if safe to do so.
   - **Alternate Routing**: Leverage alternate routes when possible to bypass congestion or incidents causing delays.
""")
st.write("""
   - **Historical Data Analysis**: Continuously analyze historical delay data to implement route-specific and time-specific interventions.
   - **Train Performance Monitoring**: Use predictive analytics to monitor train performance in real-time and issue corrective actions for late trains.
""")

# Insert figure for departure vs. arrival delays
st.image('Images/departure arrival.png', caption="Departure vs. Arrival Delay Patterns")

# 7. Predictive Modeling for ETCS Impact
st.header("7. Predictive Modeling for ETCS Impact")
st.markdown("""
Using predictive modeling, it was found that having ETCS has an **significant impact** on predicting delay times. This means that having ETCS helps us better anticipate delays, aiding train operators in improving schedules and punctuality.
(Technical Justification:
ETCS significantly impacts delay predictions, explaining 27% of the variance in delays (R-squared = 0.58). This indicates that while other factors affect delays, ETCS plays a crucial role in enhancing predictive accuracy.
""")

# Insert figure for predictive modeling
st.image('Images/feature_importance.png', caption="ETCS Impact on Delay Prediction")

# Conclusion



st.write("### Next Steps")
st.write("""
**Data Expansion**: Expand the dataset beyond a single day to cover a larger time frame (e.g., several months or years) to identify consistent patterns in train delays and incidents. This will improve the robustness of the insights.

**Explore Seasonal Variations**: Investigate seasonal effects on delays, especially during winter months, when non-ETCS lines experience significantly longer delays. This could help determine how much ETCS mitigates winter-related disruptions.

**Incident Analysis**: Perform a deeper analysis of the types of incidents causing delays (e.g., intrusions, technical failures) and evaluate the effectiveness of current mitigation strategies. Focus on Line 50A/3 and other lines with high incidents.

**Infrastructure Improvement**: Explore whether infrastructure improvements near border stations (e.g., upgrading to higher ETCS levels) could reduce delays, especially for stations with persistent high wait times.

**ETCS Optimization**: Study whether enhancing the ETCS system (e.g., upgrading from L2 FS to higher levels or implementing ETCS on non-ETCS lines) could lead to improved recovery times and reduced delays.

**Operational Adjustments**: Investigate the impact of scheduling adjustments for trains that are consistently late at departure. This could minimize cumulative delays and improve overall punctuality.

**Enhance Predictive Models**: Incorporate additional variables (e.g., weather conditions, maintenance schedules, crew shifts) to improve the accuracy of delay predictions, and develop a more comprehensive model for real-time applications. Due to lack of time only Random Forest regressor has been used, but a plethora of Machine Learning Models can be used for predictive Modelling.
         
**Identify Ripple Effect**: Separate primary and secondary delays in the dataset. This will allow for a better understanding of cascading delays. A solution can be developed to assign secondary delays to the appropriate causes, which can inform better operational decisions to minimize the ripple effect of delays.
  
**Dashboards**: Incorporating real-time data on train delays, incidents, and infrastructure conditions into the dashboard can greatly enhance decision-making. Dynamic dashboards that show live data on train movements, delays, and predicted disruptions will help railway operators manage delays more effectively.
 Develop a real-time dashboard to track delays, categorize them as primary or secondary, and predict the impact of current incidents. Real-time visualization can also inform infrastructure maintenance scheduling and provide immediate responses to reduce cascading delays.
         
**IT Maintaince**: Even though intrusions are the most common cause of delay, infrastructure-related issues (e.g., track problems, maintenance requirements) cause the most disruptive delays. These often result in severe network-wide delays.
Prioritize maintenance schedules for critical infrastructure that frequently causes delays. Analyzing data over time will allow you to predict when and where infrastructure maintenance is needed, helping to prevent future disruptions.
""")
st.image('Images/disruptive incidents.png', caption="Most Disruptive Incidents")
