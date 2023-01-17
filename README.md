# Great-Crested-Newt-UK-Pond-Analysis

The Great Crested Newt (GCN) is a European protected species and contributed many ecosystem services and to the health of our environment. The UK's district lincensing scheme handles newt protection and relocation services on behalf of developers, as previously the relocated GCN populations could end up in well intended but inappropriate "designed ponds"  and inevitably fail due to isolation or poor environmental conditons ([read here](https://freshwaterhabitats.org.uk/projects/newt-conservation/#:~:text=The%20new%20approach%20focuses%20on,newts%20can%20breed%20and%20thrive) or [watch here](https://www.youtube.com/watch?v=efJ0YYD1MbM)). 

This project intends to close the knowledge gap between ambiguous newt population sizes and ideal environmental conditions at a district level via the use of ["surveyed priority pond"](https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::surveyed-priority-ponds-england/about) data by Natural England/Defra and analysing Great Crested Newt (GCN) location patterns via the use of machine learning models

The script is intended to used via the scheduled task manager on a windows remote desktop and the output to function as an automated report (many of the visualisations below are dynamically filtered to top groups in case future datasets have numerous groups, for example). The script can be expanded to export datasets to a SQL data warehouse (coming soon) or ported directly to an Altair python dashboard

## When and when are Great Crested Newts found?

Great Crested Newt counts are summed across all years and presented below. Many sightings either did not have dates, or had poor quality (for example, many sightings are in 1905, and it is unclear if that is approriate) so these records will not be counted in the temporal visualisations

![Untitled design](https://user-images.githubusercontent.com/122735369/212549286-e11f6132-33ad-42ec-b2bb-a074f38acf66.jpg)

May, and Wednesdays, are when most GCNs are reported - however this is not indicative of the actual GCN patterns. For instance it may be that GCNS _are_ present more during this time, or it may be that the volunteers gathering the data are more active at this time. 2019 has the highest count of records, though the 2020 data is incomplete (and currently has 0 GCN sightings)

![Untitled design (6)](https://user-images.githubusercontent.com/122735369/212550950-9ac21a6f-07b3-4488-b541-c55c8d491bda.jpg)
<p align="center"><sup>Table showing spatial summary statistics, as well as occupancy proportion. Temporal data is across all sites (no records removed), and spatial data is across all years except when year ('1905') or empty dates were filtered out</sup></p>

Due to the large number of records denoting the town nearest to a survyed pond and counties in which ponds were found, spatial visualisations were filtered to the top 10 records. Note that county data was extracted seperately via the coordinates of the pond, as Natural England/DEFRA county was on an ambiguous granularity (for example - "Gloucestershire, Wiltshire and Bristol/Bath area")

![Untitled design (5)](https://user-images.githubusercontent.com/122735369/212550996-275f2d32-39c7-476c-ac56-f4d47f796300.jpg)

The following is a chloropleth map showing which counties have the highest counts of Great Crest Newts (across all years), along showing the locations all pond sites and whether they were found to occupy the site or not

![Untitled design (1)](https://user-images.githubusercontent.com/122735369/212669721-84fe39f2-2ce8-448b-8196-917dca53f4ff.png)
<p align="center"><sup>An interactive map (right) is being developed and will be available soon</sup></p>

The below table shows total GCN observations counts in each county, and is ranked by the average of all three years displayed (2017,2018, and 2019). No observer variables are available in this dataset, which can bias the dataset and subsequent predictive analytics - for example when a pond is recorded as having no GCNs, it may be that the GCN just wasn't observed, not that it wasn't present, and it is possible that certain observers are more skilled than others. The drastic population size changes seen below may be indicative of this, and highlight a need to standardise data collection methods.

![year_on_year_change](https://user-images.githubusercontent.com/122735369/212964226-8cefec1e-a4d0-4932-8b2e-d636e6a11e67.png)

## Why are they found there?

To understand drivers of GCN presence, geographical covariates were extracted from publicly available '.tif' files in a 2500m buffer around pond locations. Data was not available for all covariates across all years so the most recent geographical dataset was used. Where covariate data was not present in the buffer zone, a county wide average was used, and where a county average was not available, a county wide average was used instead. Overall, 7 geographical covariates were extracted - human footprint (consisting of cumulative human pressure consisting of 8 variables such as built up environmental and crop land scales), hillshade, elevation, human population, precipitation, windspeed 10m from the ground, and global horizontal irradiance (total solar radiation on a horizontal surface).

![Untitled design (7)](https://user-images.githubusercontent.com/122735369/212551258-9d945a85-0aea-47fe-a32b-8f498c68ca96.jpg)

<p align="center"><sup>Example visualised geographical tif datasets</sup></p>

Combinations of these variables (for example, elevation alone, or elevation with precipitation), along with a range of machine learning model architecture (such as random forest, XGboost, and logistic regression) were iteratively tested to determine which variables and model would be most accurate. Each combination was run 5 times and various performances were visualised to assist with model selection. The most accurate model (XGboost - ~71% accurate using elevation, global horizontal irradiance, human footprint, hillshade, precipitation, windspeed 10m) was then trained with various iterations, learning rates, and sampling rates to fine tune increased prediction accuracy (~76% accurate)

![Untitled design (2)](https://user-images.githubusercontent.com/122735369/212969463-2fb9d16e-3df3-42b9-85e2-f220e76166e0.png)

<p align="center"><sup>Top left: model 732 performance predicting GCN presence based on geographical data vs actual GCN presence. Top right: feature importance scores for model 732. Bottom: various performance metrics from the 5 models with the highest accuracy</sup></p>

The most important variables were indices of human development, and the amount of sunlight and rain, whereas the least important variables in this model were hillshade, windspeed and elevation. Future analyses could split apart the human footprint indices to determine the most important factor contained (for example pollution, or navigable waterways), whereas the climatic variables could be dropped seeing as England is has a temperate climate with few extremes in spatial variations.

## Where will they be found in future?

_summary: Model 732, predict site occupancy by sites across all years (reword), summed by county then compared to actual counts_
Only the positive differences where _more_ GCNs were predicted are highlighting in the right hand visual. Future analyses could include geographical data by year (as human footprint will change as new infrastructure is built, and certain years may have more sunlight) to produce a more relevant model

![Untitled design (3)](https://user-images.githubusercontent.com/122735369/212971941-19e94848-fed2-44cd-b610-88f862843398.png)

## What steps should we take?

Conservation steps:
Prioritise ponds in areas with low HFP, higher precipitation and less GHI.

Analytical steps:
data collection quality
different geographical covariates
seperate covariates for each year

