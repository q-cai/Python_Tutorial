# Learn from Lesson 10:

# Understand inputs --> Understand outputs -->
# Understand relationship -- examples --> simple mechanical solution -->
# Develop incrementally and test as we go

# Program defensively! Add assertions if the input is not valid!
# Custom for naming a variable: don't start with a caption letter


# Given your birthday and the current date, calculate your age in days.
# Account for leap days.
#
# Assume that the birthday and current date are correct dates (and no
# time travel).


daysofMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def isLeapYear(year):
    result = False
    if year % 4 == 0 and year % 100 != 0:
        result = True
    if year % 400 == 0:
        result = True
    return result


def daysInMonths(month, year):
    x = 0
    mon = 0
    for mon in range(0, month - 1):
        x += daysofMonths[mon]
        if mon == 1 and isLeapYear(year):
            x += 1
    return x


def daysBetweenDates(year1, month1, day1, year2, month2, day2):
    result = (year2 - year1) * 365 + daysInMonths(month2, year2) - \
        daysInMonths(month1, year1) + day2 - day1
    for year in range(year1 + 1, year2):
        if isLeapYear(year):
            result += 1
    return result


# Test routine

print(daysBetweenDates(1900, 1, 1, 1999, 12, 31))


def test():
    test_cases = [((2012, 1, 1, 2012, 2, 28), 58),
                  ((2012, 1, 1, 2012, 3, 1), 60),
                  ((2011, 6, 30, 2012, 6, 30), 366),
                  ((2011, 1, 1, 2012, 8, 8), 585),
                  ((1900, 1, 1, 1999, 12, 31), 36523)]
    for (args, answer) in test_cases:
        result = daysBetweenDates(*args)
        if result != answer:
            print("Test with data:", args, "failed")
        else:
            print("Test case passed!")


test()
