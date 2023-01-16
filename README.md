# Great-Crested-Newt-UK-Pond-Analysis

The Great Crested Newt (GCN) is a European protected species and contributed many ecosystem services and to the health of our environment. The UK's district lincensing scheme handles newt protected and relocation services on behalf of developers, and previously the relocated GCN populations could end up in well intended but inappropriate "designed ponds"  and inevitably fail due to isolation or poor environmental conditons ([read here](https://freshwaterhabitats.org.uk/projects/newt-conservation/#:~:text=The%20new%20approach%20focuses%20on,newts%20can%20breed%20and%20thrive) or [watch here]https://www.youtube.com/watch?v=efJ0YYD1MbM)). This project intends to close the knowledge gap between ambiguous newt population sizes and ideal environmental conditions at a district level via the use of ["surveyed priority pond"](https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::surveyed-priority-ponds-england/about) data by Natural England/Defra and analysing Great Crested Newt (GCN) location patterns via the use of machine learning models

The script is intended to used via the scheduled task manager on a windows remote desktop and the output to function as an automated report (many of the visualisations below are dynamically filtered to top groups in case future datasets have numerous groups, for example). The script can be expanded to export datasets to a SQL data warehouse (coming soon) or ported directly to an Altair python dashboard

# When and when are Great Crested Newts found?

![Untitled design](https://user-images.githubusercontent.com/122735369/212549286-e11f6132-33ad-42ec-b2bb-a074f38acf66.jpg)

_placeholder caption here? or just scrap and include below_

![Untitled design (5)](https://user-images.githubusercontent.com/122735369/212550996-275f2d32-39c7-476c-ac56-f4d47f796300.jpg)

_placeholder caption for the above_

![Untitled design (6)](https://user-images.githubusercontent.com/122735369/212550950-9ac21a6f-07b3-4488-b541-c55c8d491bda.jpg)

_placeholder caption_

![GCN_survey_chloropleth](https://user-images.githubusercontent.com/122735369/212551448-95e13a36-acdc-499c-a5af-5f8f8581b342.png)

_placeholder - something about total counts across all years for counties and pond locations [whether GCN were found or not]. some key bullet points about the method, and key takeaways - good/bad locations_

_add html interactive map? I spent ages on that... make the above smaller_

# Why are they found there?

_explain method something like 'geographical covariates were extracted in a 2500m buffer around pond locations. how many and what covs were included. where they came from_

![Untitled design (7)](https://user-images.githubusercontent.com/122735369/212551258-9d945a85-0aea-47fe-a32b-8f498c68ca96.jpg)

_placeholder caption example visualised tif datasets_

# Where will they be found in future?

_placeholder - METHODS for training best model, then fine tuning that model for the highest accuracy. explain best model, explain FIV, explain predicted vs actual performance. 

_add in image of model performances. MAYBE screenshot the iterative combination of variables as well as model testing. FIV. pred vs act_

# What steps should we take?

_placeholder RESULTS_

_image of differences and increases - explain the results_
