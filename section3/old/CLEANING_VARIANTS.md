# Cleaning Variants Guide

Input file used in all commands:
- `section3/labeled_reviews_raw.csv`

## clean_1
Meaning:
- remove URLs
- remove emojis
- remove punctuation

Output file:
- `section3/labeled_reviews_raw_clean_1.csv`

Command:
```bash
/media/abdo/Games/social_data_analysis/venv/bin/python section3/text_preprocessing_v2.py \
  --input section3/labeled_reviews_raw.csv \
  --output section3/labeled_reviews_raw_clean_1.csv \
  --remove_urls --remove_emojis --remove_punctuation
```

## clean_2
Meaning:
- remove URLs
- remove emojis
- remove punctuation
- remove stopwords (negation words like no/not are kept by your script)

Output file:
- `section3/labeled_reviews_raw_clean_2.csv`

Command:
```bash
/media/abdo/Games/social_data_analysis/venv/bin/python section3/text_preprocessing_v2.py \
  --input section3/labeled_reviews_raw.csv \
  --output section3/labeled_reviews_raw_clean_2.csv \
  --remove_urls --remove_emojis --remove_punctuation --remove_stopwords
```

## clean_3
Meaning:
- remove URLs
- remove emojis
- remove punctuation
- lemmatize text

Output file:
- `section3/labeled_reviews_raw_clean_3.csv`

Command:
```bash
/media/abdo/Games/social_data_analysis/venv/bin/python section3/text_preprocessing_v2.py \
  --input section3/labeled_reviews_raw.csv \
  --output section3/labeled_reviews_raw_clean_3.csv \
  --remove_urls --remove_emojis --remove_punctuation --lemmatize
```

## clean_4
Meaning:
- remove URLs
- remove emojis
- remove punctuation
- remove stopwords (negation words like no/not are kept)
- lemmatize text
- fix spelling

Output file:
- `section3/labeled_reviews_raw_clean_4.csv`

Command:
```bash
/media/abdo/Games/social_data_analysis/venv/bin/python section3/text_preprocessing_v2.py \
  --input section3/labeled_reviews_raw.csv \
  --output section3/labeled_reviews_raw_clean_4.csv \
  --remove_urls --remove_emojis --remove_punctuation --remove_stopwords --lemmatize --fix_spelling
```
