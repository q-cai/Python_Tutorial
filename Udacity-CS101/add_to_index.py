# Define a procedure, add_to_index,
# that takes 3 inputs:

# - an index: [[<keyword>,[<url>,...]],...]
# - a keyword: String
# - a url: String

# If the keyword is already
# in the index, add the url
# to the list of urls associated
# with that keyword.

# If the keyword is not in the index,
# add an entry to the index: [keyword,[url]]

index = []


def add_to_index(index, keyword, url):
    for item in index:
        if item[0] == keyword:
            item[1].append(url)
            return index
    return index.append([keyword, [url]])


# Test cases

add_to_index(index, 'udacity', 'http://udacity.com')
add_to_index(index, 'computing', 'http://acm.org')
add_to_index(index, 'udacity', 'http://npr.org')

print(index)

# >>> [['udacity', ['http://udacity.com', 'http://npr.org']],
# >>> ['computing', ['http://acm.org']]]


# to test again:

for item in index:
    item[0] = item[0].upper()
    item[1].append('http://www.stat.tamu.edu/~qcai/')
print(index)
