# Overview:

## Links:

**Presentation**
* [Google Slides](https://docs.google.com/presentation/d/1I34XcvTqbOwh9xBtx27wmkNFPC6luxFFKyY532R_-0I/edit?usp=sharing)
* [PDF](https://github.com/Kaewin/phase3_project3/blob/main/Phase%203%20Presentation.pdf)

**Lab Notebook (mostly unused)**
* [Google Document](https://docs.google.com/document/d/1Spref_pjFamfD-KR-_QiNYEyXASlaG7z9inboxcsCjs/edit?usp=sharing)

## Business & Data Understanding

### Business Problem:

Located in East Africa within the African Great Lakes region, Tanzania has a population of over 57,000,000 that faces significant challenges in accessing clean water. Many wells need repairs or have stopped working, while others have been added. To help address this issue, the Ministry of Water needs a classification model to identify broken wells.

I looked at three questions:

Is there a pattern in regards to:
* WHO?
    * Government
    * private business
    * etc
* WHERE
    * Hotspots
    * Patterns in location
* PERMIT STATUS
    * Does it affect probability of repair status?

### Data Source:

The Taarifa Waterpoints dashboard is an open-source platform aggregating data from the Tanzania Ministry of Water. It helps citizens stay informed about water-related issues in Tanzania, empowering them to participate in water resource management.

**Data Source**
* [Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)

# Modeling

I began with a baseline model, which had an accuracy score of 54% on the test data.

![baseline_model](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/output_40_1.png)

I then developed a decision tree model, mainly for it's speed when compared to a logistic model. 

![dtc_model](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/output_73_0.png)
![model_graph](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/output_59_3.png)

# Evaluation

When working with models that identify broken wells, it is important to prioritize minimizing false negatives. Showing up to a functioning well is worse than ignoring a broken one, which is why the Precision metric was used. The model can correctly identify negative instances and trade-offs with Accuracy by minimizing false negatives. However, this may result in identifying more false positives. The final model precision was 79%, correctly identifying 79% of broken wells as positive.

# Results:

## WHO: 

There was no clear pattern to identify individuals involved and instances were equally divided between breaking and not breaking.

![result_1](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/graph_1.png)

## WHERE: 

In Iringa, there is a high number of broken wells, which is highly correlated with the longitude/latitude. Although there are hotspots, the model itself cannot pinpoint their exact location. Other indicators of broken wells include high population and the year of construction.

![result_2](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/graph_2.png)

## Permits?

There doesn't seem to be an effect. Most wells are permitted in Tanzania, but it seems to be split down the middle both ways.

![result_3](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/graph_3.png)

## Important Result:

I want to emphasize an important finding: communal wells and hand pump wells are the waterpoints most likely to be affected. It is crucial to prioritize fixing them because they serve a large group of people and are prone to breaking down as more people use them.

![result_4](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/graph_4.png)

## Graph Of Feature Importance:

![final_graph](https://github.com/Kaewin/Predicting-Water-Wells-In-Need-Of-Repair-For-The-Government-Of-Tanzania/blob/main/images/output_79_0.png)

# Conclusion

One of the biggest problems with the data is that there are overlapping categories, so that it could use some cleaning up. Additionally, managing multicollinearity is proving to be very difficult. It may be helpful to look into hotspots where certain places are more likely than others to have issues. For example, Iringa could be an excellent place to focus on, specifically looking into communal standing pipes, handpumps, pumps with low water, old equipment, and a high population.
