{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 LucidaGrande;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww29200\viewh15800\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # DIC Phase 2 - Project Overview\
\
This project involves **large-scale data ingestion, cleaning, exploratory analysis, and machine learning modeling** using **PySpark** and a **Docker-based Hadoop** cluster.\
\
---\
\
## Project Structure\
\
### `/Code/`\
Contains all Python scripts organized by stage:\
\
| File | Purpose |\
|:---|:---|\
| `01_load_csv_to_parquet.py` | Load raw CSV files and save as Parquet into HDFS. |\
| `02_booksdata_cleaning_eda.py` | Perform EDA and cleaning on the `books_data.csv` dataset. |\
| `03_reviews_cleaning_eda.py` | Perform EDA and cleaning on the `Books_rating.csv` dataset. |\
| `04_merge_eda.py` | Merge cleaned datasets, plot visualizations, and save merged dataset. |\
| `05_data_analysis.py` | Execute all five major data analysis objectives with visual outputs. |\
| `06_ml_objective3.py` | Perform Reader Segmentation and Clustering (KMeans + PCA). |\
| `07_ml_objective2.py` | Conduct Sentiment Analysis on book reviews (TextBlob, WordClouds). |\
| `08_ml_objective1.py` | Build Regression Model for Review Score Prediction and ALS Recommender System. |\
\
---\
\
### `/Output/`\
Stores results generated from different analysis and machine learning tasks:\
\
#### `/Output/Plots/`\
Organized into subfolders:\
- **EDA Plots** 
\f1 \uc0\u8594 
\f0  Basic Exploratory Data Analysis plots.\
- **Data Analysis Plots** 
\f1 \uc0\u8594 
\f0  Visuals supporting the 5 data analysis objectives.\
- **ML Problem Plots** 
\f1 \uc0\u8594 
\f0  Visualizations for clustering, regression, and recommendation tasks.\
\
#### `/Output/Summaries/`\
Plain-text summaries (`.txt` files):\
\
| File | Contents |\
|:---|:---|\
| `books_eda_cleaning_summary.txt` | Summary of books dataset EDA and cleaning. |\
| `reviews_eda_cleaning_summary.txt` | Summary of reviews dataset EDA and cleaning. |\
| `merged_cleaning_summary.txt` | Summary of merged dataset cleaning and inspection. |\
| `sentiment_summary_obj2.txt` | Sentiment analysis report (Objective 2). |\
| `ml_obj3_summary.txt` | Reader clustering and segmentation report (Objective 3). |\
\
---\
\
## HDFS Directory Layout\
\
All important intermediate and final outputs are saved into **HDFS**, organized under `/user/n1/parquet/`:\
\
\
## Dockerized Hadoop Setup\
\
The Hadoop environment was deployed using Docker containers:\
\
| Container | Purpose |\
|:---|:---|\
| `namenode` | HDFS NameNode |\
| `datanode` | HDFS DataNode |\
| `resourcemanager` | YARN Resource Manager |\
| `nodemanager` | YARN Node Manager |\
| `freshone` | Main container for executing PySpark jobs |\
\
Screenshots of the running containers and HDFS file browser are included in the final project report for reference.\
\
}