# Overdose Deaths Analysis Over COVID-19

## Project Description

**Research Question:** "Does lower income across boroughs have more overdose related deaths than their higher income counterparts?"

I thought it would be interesting to explore the cause of deaths in NYC. Having representation in the forgotten borough where drugs run rampant, it is an enduring issue I would love to get my hands on.  Specifically one of irresponsibly use, such as opioid overdosing. But I could not just stop there. My data would not be complete unless I had assessed all the boroughs respectively. This project aims to examine certain trends that come in relation with respective boroughs and incomes, up until the peak of the pandemic.

## Project Expectations/Objectives

When this project is completed, we are hoping to analyze overdose deaths that have occurred across a five year time span (Pre to peak pandemic era).

This information is insightful because it makes us want to understand a direct correlation between income situation and the cause of overdose deaths. One may be able to infer certain hypotheses, but the data will also show either how skewed or how close these deaths are alongside each other, allowing us to assess the true significance.

We also plan on streamlining it across a pseudo-bargraph with "matplotlib". That way the research can be presentable to technical and non technical individuals alike.

**Goals:**
- To assess each boroughs death rate and borough income alongside many types of graphs: like lines and bars to better understand and visualize correlations and trends with income by borough and drug overdosing death rate.

## Data Sources

### Income Data
I retrieved my initial data on the relative incomes of each borough from https://data.census.gov/. I specifically searched for the "U.S. Census Bureau, American Community Survey (ACS) 5-Year Estimates, 2018-2022". This gave me a wide scope of what income was relatively like in each borough, as well as the zipcodes that essentially helped showcase how economically diverse each borough is.

- **URL:** https://data.census.gov/
- **File Type:** CSV

### Overdose Death Data
Aside from getting the income sources in each borough, we of course had to follow up with the number of overdose deaths per borough. Over the same, if not similar timespan. This part was considerably difficult, taking me a few hours, as there were not well documented sources of opioid/drug overdosing/poisoning. So I had to use a plethora of assets from the Epiquery dataset, and then I used AI to extract and help organize the file so that it listed each overdose year by year per borough.

**Data Sources by Year:**
- **2022 Data:** https://www.nyc.gov/assets/doh/downloads/pdf/epi/databrief137.pdf
- **2021 Data:** https://www.nyc.gov/assets/doh/downloads/pdf/epi/databrief133.pdf
- **2020 Data:** https://www.nyc.gov/assets/doh/downloads/pdf/epi/databrief131.pdf | https://www.nyc.gov/assets/doh/downloads/pdf/epi/databrief129.pdf
- **2019 Data:** https://www.nyc.gov/assets/doh/downloads/pdf/epi/databrief122.pdf
- **2018 Data:** https://www.nyc.gov/assets/doh/downloads/pdf/epi/databrief116.pdf

Similarly to our last post, the CSVs will be added here for ease of use. The data is practically the same as before, but molded differently and experientially to express better outcomes. All the while maintaining credibility.

## Updates/Change Notes

- Removed the Poverty Rate chart as it overlapped with the similar comparison of "income vs overdose deaths".
- Removed the heatmap due to limit redundancy as well as to prevent overstimulation of too many graphs being presented.
- Fixed the bubble sizes in the legends so that they showcase better consistency. (Thanks, to my humble friend Steven Lau)! (Done offline :p)

This project helps familiarize myself with data analytics in Python, and how it can be used to assess real and ongoing world issues. I found this very exciting to do. 7.5 smiles out of 10. 
