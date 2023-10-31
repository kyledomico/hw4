# Implement a character-based Naive Bayes classifier that classifies a document as English, Spanish, or Japanese -
# all written with the 26 lower case characters and space.
# The dataset is languageID.tgz, unpack it. This dataset consists of 60 documents in English, Spanish and Japanese.
# The correct class label is the first character of the filename: y ∈ {e, j, s}. (Note: here each file is a document in
# corresponding language, and it is regarded as one data.)

# We will be using a character-based multinomial Na ̈ıve Bayes model. You need to view each document as a bag of
# characters, including space. We have made sure that there are only 27 different types of printable characters (a to
# z, and space) – there may be additional control characters such as new-line, please ignore those. Your vocabulary
# will be these 27 character types. (Note: not word types!)

# 1. Use files 0.txt to 9.txt in each language as the training data. Estimate the prior probabilities ˆp(y = e), ˆp(y =
# j), ˆp(y = s) using additive smoothing with parameter 1
# 2 . Give the formula for additive smoothing with
# parameter 1/2 in this case. Print and include in final report the prior probabilities. (Hint: Store all probabilities
# here and below in log() internally to avoid underflow. This also means you need to do arithmetic in log-
# space. But answer questions with probability, not log probability.)

# Read the data files
# Create a list of all the files
# Create a list of all the labels
def get_text(file):
    # Read Contents
    with open(file, 'r') as f:
        text = f.read()
    
    # Remove new line characters
    text = text.replace('\n', ' ').strip()

    # Remove any double or triple spaces
    text = text.replace('    ', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')

    return text

# Generate Dataset of all text files as features and labels as first letter of filename
training_data = []
training_labels = []

for lang in ['e', 'j', 's']:
    for i in range(10):
        file = f'languageID/{lang}{i}.txt'
        training_data.append(get_text(file))
        training_labels.append(lang)

# Generate the Prior Probabilities with additive smoothing of 1/2 and Print them
prior_probabilities = {}
for lang in ['e', 'j', 's']:
    prior_probabilities[lang] = (training_labels.count(lang) + 0.5) / (len(training_labels) + 0.5*3)

# Store Log Probabilities
import math
log_prior_probabilities = {}
for lang in ['e', 'j', 's']:
    log_prior_probabilities[lang] = math.log(prior_probabilities[lang])

print('Prior Probabilities:')
print(prior_probabilities)

# 2. Using the same training data, estimate the class conditional probability (multinomial parameter) for English
# θ_i,e := ˆp(c_i | y = e)
# where c_i is the i-th character. That is, c_1 = a, . . . , c_26 = z, c_27 = space. Again use additive smoothing
# with parameter 1/2 . Give the formula for additive smoothing with parameter 1/2 in this case. Print θ_e and
# include in final report which is a vector with 27 elements.

# Generate the Class Conditional Probabilities for English with additive smoothing of 1/2 and Print them
c_i = {}
for i in range(27):
    c_i[i] = chr(ord('a') + i)
c_i[26] = ' '

class_conditional_probabilities = {}
for lang in ['e', 'j', 's']:
    class_conditional_probabilities[lang] = {}
    # Select indices of training data that are in the language
    indices = [i for i, x in enumerate(training_labels) if x == lang]
    # Get all the text in the language
    text = ''.join([training_data[i] for i in indices])
    for i in range(27):
        class_conditional_probabilities[lang][i] = (text.count(c_i[i]) + 0.5) / (len(text) + 0.5*27)

# Print the 'e' probabilities as a vector
print('Class Conditional Probabilities for English:')
print(class_conditional_probabilities['e'])

# 3. Print θj , θs and include in final report the class conditional probabilities for Japanese and Spanish.

# Print the 'j' probabilities as a vector
print('Class Conditional Probabilities for Japanese:')
print(class_conditional_probabilities['j'])

# Print the 's' probabilities as a vector
print('Class Conditional Probabilities for Spanish:')
print(class_conditional_probabilities['s'])

# 4. Treat e10.txt as a test document x. Represent x as a bag-of-words count vector (Hint: the vocabulary has
# size 27). Print the bag-of-words vector x and include in final report.

# Generate the bag-of-words vector for e10.txt
test_file = 'languageID/e10.txt'
test_data = get_text(test_file)
test_vector = {}
for i in range(27):
    test_vector[i] = test_data.count(c_i[i])

# Print the bag-of-words vector for e10.txt
print('Bag-of-Words Vector for e10.txt:')
print(test_vector)

# 5. Compute ˆp(x | y) for y = e, j, s under the multinomial model assumption, respectively. Use the formula
# ˆp(x | y) =
# d∏
# i=1
# θxi
# i,y
# where x = (x1, . . . , xd). Show the three values: ˆp(x | y = e), ˆp(x | y = j), ˆp(x | y = s). Hint: you may
# notice that we omitted the multinomial coefficient. This is ok for classification because it is a constant w.r.t. y.

# Compute the probability of the test vector for each language
log_probabilities_test = {'e': 0, 'j': 0, 's': 0}

for lang in ['e', 'j', 's']:
    for i in range(27):
        log_probabilities_test[lang] += math.log(class_conditional_probabilities[lang][i]) * test_vector[i]

# Print the log probabilities of the test vector for each language
print('Log Probabilities of the test vector for each language:')
print(log_probabilities_test)

# 6. Use Bayes rule and your estimated prior and likelihood, compute the posterior ˆp(y | x). Show the three
# values: ˆp(y = e | x), ˆp(y = j | x), ˆp(y = s | x). Show the predicted class label of x.

# Compute the posterior probability of the test vector for each language
log_posterior_probabilities = {'e': 0, 'j': 0, 's': 0}
for lang in ['e', 'j', 's']:
    log_posterior_probabilities[lang] = log_prior_probabilities[lang] + log_probabilities_test[lang]

# Print the log posterior probabilities of the test vector for each language
print('Log Posterior Probabilities of the test vector for each language:')
print(log_posterior_probabilities)

# 7. Evaluate the performance of your classifier on the test set (files 10.txt to 19.txt in three languages). Present
# the performance using a confusion matrix. A confusion matrix summarizes the types of errors your classifier
# makes, as shown in the table below. The columns are the true language a document is in, and the rows are
# the classified outcome of that document. The cells are the number of test documents in that situation. For
# example, the cell with row = English and column = Spanish contains the number of test documents that are
# really Spanish, but misclassified as English by your classifier.

# Generate Dataset of all text files as features and labels as first letter of filename
test_data = []
test_labels = []

for lang in ['e', 'j', 's']:
    for i in range(10, 20):
        file = f'languageID/{lang}{i}.txt'
        test_data.append(get_text(file))
        test_labels.append(lang)

# Generate the bag-of-words vectors for the test data
test_vectors = []
for i in range(len(test_data)):
    test_vector = {}
    for j in range(27):
        test_vector[j] = test_data[i].count(c_i[j])
    test_vectors.append(test_vector)

# Compute the posterior probability of the test vectors for each language
log_posterior_probabilities = []
for i in range(len(test_vectors)):
    log_posterior_probabilities.append({'e': 0, 'j': 0, 's': 0})
    for lang in ['e', 'j', 's']:
        for j in range(27):
            log_posterior_probabilities[i][lang] += math.log(class_conditional_probabilities[lang][j]) * test_vectors[i][j]
        log_posterior_probabilities[i][lang] += log_prior_probabilities[lang]

# Compute the predicted labels of the test vectors
predicted_labels = []
for i in range(len(test_vectors)):
    predicted_labels.append(max(log_posterior_probabilities[i], key=log_posterior_probabilities[i].get))

# Compute the confusion matrix and plot it with seaborn and save to 'confusion_matrix.pdf'
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

confusion_matrix = pd.crosstab(pd.Series(predicted_labels, name='Predicted'), pd.Series(test_labels, name='Actual'))
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.savefig('confusion_matrix.pdf')
