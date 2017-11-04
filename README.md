# Apartment Interest Prediction
Predict people interest in renting specific apartments. The challenge combines structured data, geolocalization, time data, free text and images.


## Overview of my solution

This solution features Gradient Boosted Trees (`XGBoost` and `LightGBM`) and does not use stacking, due to lack of time.

### Feature engineering
Features can be activated and deactivated by a single comment in main.py

#### Time features
From the datetime field I created several features:

* Year, month, day, hour, day of the week
* Bank holiday, school holiday
* Elapsed time since publication

Furthermore, day, month, hour are cyclical.
To tell the classifier than after Sunday (day 6) there is Monday (day 0), I've projected the time information on a circle by taking the cos and sin.

#### Geo-localization features
From the latitude and longitude, I created clusters using Density-based clustering ([HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)).

I would have preferred `DBSCAN` and setting epsilon to 200 meters but unfortunately, `Scikitlearn`'s `DBSCAN` is not properly optimized. Trying to get 40000 (train set) or 70000 (test set) pairwise haversine distance goes KABOOM on my memory.
(HDBSCAN creates cluster fully automatically from density, but NYC is too dense)

From the public kernels I've also taken the coordinate of Central Park, Brooklyn, Queens .... to compute the distance of each apartment from those center.

#### Apartment features
Apartment features (cat, dog, doorman, laundry in building ...) were deduplicated and encoded using a 4-letter encoding scheme to reduce duplication further.
Furthermore Sklearn CountVectorizer to One-Hot-Encode + Expose their frequency to the classifier

#### Description features (NLP / Text-mining)
The description field was one of my big focus, I did:
 - Clean-up the HTML tags from the description with `BeautifulSoup`
 - Latent Semantic Analysis, by stacking a LightGBM kernel on a description preprocessed with TfIdf + TruncatedSVD
 - Sentiment Analysis with `TextBlob` (unused at the end)
 - Extraction of metro lines and metro/transport related vocabulary
 - Check the number of words and lines
 - Check the presence of "REDACTED"
 - Check the number of caps and exclamation points

#### Categorical features
On price, number of bathrooms, bedrooms, the usual combinations of price per room, etc were done.
Address, manager, building id were numerically encoded.

Furthermore for manager and building id, various other encoding scheme were tested (Bayesian target label encoding, low/mid/high interest count from the Kaggle Forum, manager skill and building hype).

In the end, after multiple leaks on cross-validation, I simply binned managers/building with their frequency (top 1%, 2%, 5% ...).
This way target labels were not used, I ensure no leak and performance seemed to be similar to Bayesian encoding.

#### Outliers removal
Detected Outliers were corrected from the test set (117 bathrooms :O)
Prices > 13000 were clipped

#### Images
Like many other I didn't process the image at ll, besides using the magic leak (folder creation time).
The biggest issue was that the number of images per apartment was irregular, some had a floor plans, other had furnitures, other had nothing.

I did extract metadata from the images to process add resolution, image height and width to my model.
Unfortunately the json file was 800MB or 1.4GB in CSV with thousands of sparse columns. Pandas couldn't load that in my machine. The workaround would be to a. buy more RAM, b. use a dictionary structure but it was clunky and time consuming.

Example metadata are available in my 000_Data_Exploration.ipynb notebook.

## Overview of the architecture
I ran early in scalability issues and cross-validation issues with `Scikit-Learn`.

In Sklearn, you can use Pipelines to apply modifications on the train and test set independently,
but it's not trivial to use pipelines on a validation set (split from train set) that you will use as input for `XGBoost`or `LightGBM` early stopping.
Furthermore, most features are not inherently leaky and do not need to be recomputed for each fold as `Sklearn` does.
Lastly, `Sklearn` has no caching framework

So:

I wrote my own code so that adding each features is easy and independant, check the `star_command.py` pipe function.
Now each transformation can be applied with:
```Python
# Feature extraction - sequence of transformations
tr_pipeline = feat_extraction_pipe(
    tr_remove_outliers,
    tr_numphot,
    tr_numfeat,
    tr_numdescwords,
    tr_desc_mining,
    tr_datetime,
    tr_split_bath_toilets,
    tr_tfidf_lsa_lgb)
```

Feature selection was done the same way, with a framework that can deal with dataframe and sparse array, there is even a glimpse of feature selection on multiple processes, but it was slower due to Python's Global Interpreter Lock
Each features can be chained with Scikit's transformers like TfIdf or PCA.
Multiple features can be declared at the same time.

```python
select_feat = [
  ("dedup_features", CountVectorizer(max_features=200)),
  ("description", [TfidfVectorizer(max_features=2**16,
                         min_df=2, stop_words='english',
                         use_idf=True),
                TruncatedSVD(2), # 2 or 3
                # Normalizer(copy=False) # Not needed for trees ensemble and Leaky on CV
                ]),
  #("description",[HTMLPreprocessor(),NLTKPreprocessor(),
  #                TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)]
  #),
  ("description", CountVectorizer(vocabulary=vocab_metro,binary=True)),
  ("description", CountVectorizer(vocabulary=vocab_metro_lines,binary=True, lowercase=False)),
  ("redacted", None),
  (['top_' + str(p) + '_manager' for p in [1,2,5,10,15,20,25,30,50]],None)
  (['top_' + str(p) + '_building' for p in [1,2,5,10,15,20,25,30,50]],None)
  ]
```

Each transformation can be cached in a "database" with `shelve` and retrieved easily with a key. See `transformers_nlp_tfidf.py`
And finally I wrote my own cross-validation and out of fold prediction code.

Thank you for your attention
