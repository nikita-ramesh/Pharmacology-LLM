IUPHAR Queries Training Set
===========================

_Simon Harding and David Sterratt, 21 December 2024_

The file `all_queries_categorised_train.csv` contains 51 natural
language queries and associated SQL queries prepared by Simon Harding
and colleagues.

It is a subset of all the queries the group generated; the remaining
32 queries have been reserved for a test set.

The queries are classified by difficulty as assessed by Simon Harding,
and indicated by the columns below. When constructing the
training/test split, we stratified so that there are approximately
equal proportions of easy, easy-moderate, moderate-hard and hard
queries in each set. We have also tried to ensure that when there are
groups queries that we regard as similar, they are split between the
training and test set.

We regard that some natural language queries are vague and have no
right answer - in these cases the SQL is an example of a good answer.

The meaning of the columns is as follows:

- `ID` - The ID of the query in the full list of queries.

- `Difficulty: Easy`, `Difficulty: Easy-Moderate`, `Difficulty:
   Moderate-Hard`, `Difficulty: Hard` - There is in a '1' in the
   column to indicate the level of difficulty.

- `Greek` - Natural language query contains Greek letter

- `Vague/No definite right answer` - If the output is likely to be vague.

- `Minimum output columns` - indicates the minimum columns that would
   need retrieved to provide an answer - the majority need only one,
   but in most of those cases bringing back only one would not be as
   useful to the user.

- `Notes for student` - Simon's notes about aspects of the schema.

- `Training/test set` - If the item is in the training or test set

- `SQL` - The SQL query corresponding to the natural language query

- `2nd SQL` - A second possible SQL query (filled in for only a few queries).

