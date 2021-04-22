# Know-Thy-Customer
Customer segmentation using K-means clustering and dimensionality reduction. 
![](/images/Customer-Segmentation-fp.jpg)

Clever marketers understand that a company’s  best customers are their **existing customers**.  Offering tailor-made solutions to existing clients can strengthen the company’s relationship with a client by showing them that they ‘**know**’ them and ‘**get**’ what they are all about.  

## Customer Segmentation
Customer segmentation is used by marketing teams to sort customers into groups based on their demographics, behaviour, interests and/or location. 

This divides users into similar or homogeneous groups that exhibit similar traits.  Once these groups are known to the marketing team they can create products that speak to the specific needs of each group. 

## Machine Learning meets the Marketing Team

Machine learning provides the modern marketer with a number of algorithms, tools and techniques to analyse large amounts of data and reveal patterns, correlations, differences and similarities that might be hidden to the naked eye. Clustering algorithms belong to a family of unsupervised learning techniques that does just this.

In this project I will perform a **customer segmentation** on the [Credit Card Dataset](https://www.kaggle.com/arjunbhasin2013/ccdata) from the [Kaggle](https://www.kaggle.com) website using clustering and dimensionality reduction techniques interchangibly. 

## A Look At The Data

The [Credit Card Dataset](https://www.kaggle.com/arjunbhasin2013/ccdata) is a summary of the usage patterns of around 9,000 customers collected by the bank (our client) over a 6 month period.  The dataset measures 18 different features that describe each client and their interaction with their credit facility. More details on these features are available later in the notebook.

A quick look at the data dictionary reveals the following information about each feature in the dataset:

  1.  **CUSTID** : Identification of Credit Card holder (Categorical)

  2.  **BALANCE** : Balance amount left in their account to make purchases

  3.  **BALANCE_FREQUENCY**  : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)

  4.  **PURCHASES** : Amount of purchases made from account

  5.  **ONEOFF_PURCHASES** : Maximum purchase amount done in one-go

  6.  **INSTALLMENTS_PURCHASES** : Amount of purchase done in installment

  7.  **CASH_ADVANCE** : Cash in advance given by the user

  8.  **PURCHASES_FREQUENCY** : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)

  9.  **ONEOFF_PURCHASES_FREQUENCY** : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)

  10.  **PURCHASES_INSTALLMENTS_FREQUENCY** : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)

  11.  **CASH_ADVANCE_FREQUENCY** : How frequently the cash in advance being paid

  12.  **CASH_ADVANCE_TRX** : Number of Transactions made with "Cash in Advanced"

  13.  **PURCHASES_TRX** : Numbe of purchase transactions made

  14.  **CREDIT_LIMIT** : Limit of Credit Card for user

  15.  **PAYMENTS** : Amount of Payment done by user

  16.  **MINIMUM_PAYMENTS** : Minimum amount of payments made by user

  17.  **PRC_FULL_PAYMENT** : Percentage of full payments paid by user

  18.  **TENURE** : Tenure of credit card service for user

**Data Source: https://www.kaggle.com/arjunbhasin2013/ccdata**


## Methodology Overview

I will attempt **three different approaches**  to clustering the data and compare the outputs to get to the ideal customer clusters.  The first will be a clustering on the full feature dataset.  In the second and third approaches I will perform dimension reduction and the apply clustering on the reduced dataset. 


## Definitions

### Clustering

> Clustering is a method of **unsupervised learning** that finds similarities and patterns in data points and groups them together accordingly.  This is done without prior knowledge of the data features. 

### K-Means

> **K-means** is one of the best known clustering algorithms in machine learning. It creates *k* homogeneous groups by minimising the Euclidian distance between the data points in a cluster. The number of clusters, *k*, must be provided as an argument. 



> **K-means clustering** does have 2 very important assumptions that can easily ruin your analysis if not taken into account.  These assumptions are:

> 1.   Clusters are spatially grouped or spherical
2.   Clusters are of similar size

### Dimensionality Reduction

> Real life datasets often contain huge numbers of features.  This might make it complicated to extract homogeneous groups from the data. 

> There are several ways, automated and manual, that we can reduce the dimensions of the data to get a better handle on what cohorts exist within a dataset. 

### Principal Component Analysis

> **PCA** is an unsupervised machine learning algorithm that performs dimensionality reduction while attempting at keeping the original information unchanged. PCA works by finding composites of features called components. 

### RFM Analysis

> **RFM Analysis** is an easy and effective segmentation technique that divides types of data into recency, frequency and monetary data bundles and works out a score for each.

> Once the data is segmented this way, algorithms can be performed on the newly formed dataset.

> This technique combines knowledge of the data and the environment in which the data exists with the power of machine learning algorithms. 

### Method 1 : K-Means and the Elbow Method on Full Feature Set

In this evaluation we ran **K-means** multiple times and using the **Elbow method** determined that k=8 is the optimal clustering solution. 
![Elbow Graph](/images/m1-elbow.png)
![](/images/m1-clusters.png)
![](/images/m1-hist.png)

Evaluating the clusters we see that the data in each cluster is more normally distributed than what was observed in the first histograms.  Although these visualisations contain a wealth of information for the marketing team to mine, analyse and use, there is a lot of overlapping and it is not really presenting us with a clear picture of each of the clusters. 

In the next method we will see if we can improve on this set of clusters.

## Shout-outs and References

This project was inspired by a notebook from a [Udemy courses](https://www.udemy.com) by respected [Prof Ryan Ahmed Ph.D, MBA](https://www.udemy.com/user/ryan-ahmed/) on clustering. 

The dataset is not perfect for a RFM since it contains not dates. To solve the **missing data** problem for the **RFM Analysis**, I came across a [comprehensive article](https://towardsdatascience.com/customer-segmentation-using-the-instacart-dataset-17e24be9c0fe) written by JR Kreiger on customer segmentation and how to adapt less than perfect data for a RFM analysis. A huge shout-out for the article, lots of good references and linked code.

I was also able to glean helpful insights into the **practical application of k-means** clustering by reading through [Burak Özen](https://towardsdatascience.com/customer-journey-based-segmentation-for-marketplaces-70e5a56838a7)'s article on K-means and it's limitations. 




In **Method 2** we will first reduce the dimensionality of the dataset to 10 features by using the encoded layer of an **Autoencoder artificial neural network**.  

We will then group the **smaller dataset** with **K-means** at different *k-values* and apply the **Elbow method** to find the **optimal k**. 

Once again we will create visual representations of the results to evaluate and compare. 

For **Method 3** we will perform a **RFM Analysis** (recency, frequency and monetary).  This will require us to first categorise the data features into recency-, frequency- and monetary groups.  Once grouped we will work out a score for each category.   

Next we will use the three scores, recency-score, frequency-score and monetary score, run **K-means**, apply the **Elbow method** to find optimal k.  Following this we will again **visualise the clusters**, then **visualise the features** and attempt to **describe** each of the clusters. 

Since our dataset does not have any transaction dates in it, we will have to improvise to find a way to represent the **recency** metric. 

  

 






