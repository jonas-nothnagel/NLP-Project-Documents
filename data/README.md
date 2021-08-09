# Data 

## 1. raw
This folder contains all the raw, unedited data. It contains the most recent data dump from the API data pipeline which is automatically replaced when running the data collection code. It further contains all versions of the taxonomy that we produced throughout the process. The highest number indicates the latest one. It also contains the indicators for each project.

## 2. interim
The interim data folder contains all data that is transformed and processed throughout the data processing pipeline. Here we have most importantly the interim processed portfolio data from the PIMS+ API as well as the interim processed taxonomy.

## 3. processed
This folder contains the final processed data sets that are used for model training and data analysis. Here we have the merged data from the API and the excel sheets and both training data as well as labels are ready to be fed into models. There are many files for different purposes but the common demoninator is that they do not require any more cleaning, transformation, processing.

## 4. temp
This folder stores data that is temporary and produced throughout experimentations. It can be savely deleted once in a while. 