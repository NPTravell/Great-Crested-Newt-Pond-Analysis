<p align="right"><sup>Latest record on: 20/12/2020<br /> 
Data last refreshed on: 05/04/2022<br /> 
Data last analysed on: 16/01/2023</sup></p>


# Great-Crested-Newt-UK-Pond-Analysis

The UK has a 'district licensing scheme' that handles Great Crested Newt (GCN) protection and relocation services on behalf of developers ([read here](https://freshwaterhabitats.org.uk/projects/newt-conservation/#:~:text=The%20new%20approach%20focuses%20on,newts%20can%20breed%20and%20thrive) or [watch here](https://www.youtube.com/watch?v=efJ0YYD1MbM)), however there is still much to learn about about the distribution of GCN populations. 

This Python project intends to close the knowledge gap between newt population sizes and ideal environmental conditions by analysing Great Crested Newt (GCN) location data ["surveyed priority pond"](https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::surveyed-priority-ponds-england/about) provided by Natural England/Defra

## When and when are Great Crested Newts (GCN) found?


May, and Wednesdays, are when most GCNs are reported - this isn't neccessarily when they're most active - it might just be when volunteers are carrying out data collection.

![Screenshot 2024-06-03 132054](https://github.com/NPTravell/Great-Crested-Newt-Pond-Analysis/assets/122735369/e64e2125-079d-4593-b49d-598431cc84f7)


The following is a chloropleth map showing which counties have the highest counts of GCN (across all years), along with showing the locations all pond sites and whether GCN were found to occupy the site or not

![Untitled design (1)](https://user-images.githubusercontent.com/122735369/212669721-84fe39f2-2ce8-448b-8196-917dca53f4ff.png)
<p align="center"><sup>An interactive map (right) is being developed and will be available soon</sup></p>

No data collection meta data is available in this dataset; potentially biasing the dataset and subsequent predictive analytics - for example when a pond is recorded as having no GCNs, it may be that the GCN just wasn't observed, not that it wasn't present, as certain volunteers may be more skilled than others

## Why are Great Crested Newts (GCN) found where they are?

To understand drivers of GCN presence, geographical covariates were extracted from publicly available '.tif' files (to the closest year available) in a 2.5km buffer around pond locations. Where there wasn't any data for the buffer zone, a district wide average was used

The following 7 geographical covariates were used:
- [human footprint](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint)
- hillshade
- elevation
- human population
- precipitation
- windspeed 10m from the ground
- horizontal irradiance (total solar radiation on a horizontal surface)

![Untitled design (7)](https://user-images.githubusercontent.com/122735369/212551258-9d945a85-0aea-47fe-a32b-8f498c68ca96.jpg)

<p align="center"><sup>Example visualised geographical tif datasets. Each pixel corresponds to a covariate value, which was extracted at each pond site</sup></p>

Iterative machine learning model architecture (for example: random forest, XGboost, and logistic regression) and variable inclusion (such as one specific variable alone, 2 specific variables, etc.) was tested.

Model 732 (using "XGboost" architecture and elevation, global horizontal irradiance, human footprint, hillshade, precipitation, and windspeed data) was chosen due to its consistent performance, mean accuray performanc, and low run time. The was then fine tuned to acheive increased prediction accuracy.

![Untitled design (2)](https://user-images.githubusercontent.com/122735369/212969463-2fb9d16e-3df3-42b9-85e2-f220e76166e0.png)

<p align="center"><sup>Top left: model 732 performance predicting GCN presence based on geographical data vs actual GCN presence. Top right: feature importance scores for model 732. Bottom: various performance metrics from the 5 models with the highest accuracy</sup></p>

The most important variables were:
- indices of [human pressure](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint)
- sunlight
- rain

Whereas the least important variables in this model were:
- hillshade
- windspeed
- elevation

## Where will Great Crested Newts (GCN) be found in future?

Model 732 was used to predict occupancy at each site and summed each district. As a result, we have potential leads on where we can find potentially hidden GCN populations, and can drive more targeted volunteer recruitment.

![Untitled design (4)](https://user-images.githubusercontent.com/122735369/213144445-a50c484f-54dc-4c97-ac6e-368c35221cbf.png)

