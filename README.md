# Infrabel Train Delay Analysis and ETCS Impact Report

This repository contains an analysis of train delays and incidents data from Infrabel, with a focus on the impact of the European Train Control System (ETCS). The project provides both a comprehensive analysis of train delays and the impact of incidents, and it demonstrates the effects of ETCS on punctuality and recovery times.

The analysis includes:
- Exploratory Data Analysis (EDA) with visualizations.
- Predictive modeling to assess the influence of ETCS on delays.
- A Streamlit dashboard that provides an interactive report.
  
All CSV datasets used in the analysis are provided in the repository, and all visualizations can be reproduced using the code in `main.py`. Additionally, pre-generated visualizations are included for easy setup of the Streamlit app.

## Repository Structure

```bash
Infrabel-Data-Analysis/
│
├── __pycache__/                # Python cache files
├── myvenv/                     # Virtual environment
├── Data                        # All required datasets
├── analysis.py                 # Python file for analysis functions
├── preprocess.py               # Python file for data preprocessing functions
├── visualization.py            # Python file for visualizations
├── main.py                     # Main analysis script
├── streamlit.py                # Streamlit app script
├── Infrabel_Data_Analysis.ipynb # Jupyter Notebook with analysis
├── requirements.txt            # Python dependencies
├── Images/*.png                       # Pre-generated visualizations (for Streamlit app)
    └── *.html                      # Pre-generated HTML maps for the Streamlit app
```

## Getting Started

### 1. Set Up a Virtual Environment

To get started with the project, you need to set up a Python virtual environment. Follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shreyab375/Infrabel-Data-Analysis.git
   cd Infrabel-Data-Analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv myvenv
   ```

3. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     myvenv\Scripts\activate
     ```
   - On **MacOS/Linux**:
     ```bash
     source myvenv/bin/activate
     ```

### 2. Install Required Dependencies

After activating the virtual environment, install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Running the Python Code

You can execute the main analysis by running `main.py`. This will preprocess the datasets, perform analysis, generate visualizations, and build a predictive model.

```bash
python main.py
```

### 4. Running the Streamlit App

The interactive Streamlit app provides a report summarizing the results of the analysis with visualizations. To launch the app, execute:

```bash
streamlit run streamlit.py
```

This will open a web browser with the dashboard that includes:
- Visualizations showing delays, incidents, and ETCS impact.
- A summary of the key findings.
- Interactive maps and charts.

### 5. Using the Jupyter Notebook

For a step-by-step walkthrough of the analysis, you can also check the Jupyter notebook:

```bash
jupyter notebook Infrabel_Data_Analysis.ipynb
```

## Key Features

- **Delays Analysis**: Provides insights into the relationship between delays, stopping places, and train lines.
- **ETCS Impact**: Demonstrates the benefits of ETCS in improving recovery times after incidents.
- **Incident Hotspots**: Identifies areas prone to incidents and their impact on delays.
- **Predictive Modeling**: Assesses the influence of different factors on train delays, including ETCS.

## Datasets

- `Regular data.csv`: Contains regular train delay data.
- `Incidents.csv`: Incident data affecting train operations.
- `ETCS.csv`: Data regarding the deployment of ETCS across different lines.

## Pre-Generated Visualizations

The repository includes several pre-generated visualizations (`*.png` and `*.html` files) to streamline the deployment of the Streamlit app. These visualizations include maps, charts, and figures illustrating the key insights from the analysis.


