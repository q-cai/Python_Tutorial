# A different way to solve this problem
# by adopting the idea in Lesson 10
#
# Use Dave's suggestions to finish your daysBetweenDates
# procedure. It will need to take into account leap years
# in addition to the correct number of days in each month.

daysofMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def isLeapYear(year):
    result = False
    if year % 4 == 0 and year % 100 != 0:
        result = True
    if year % 400 == 0:
        result = True
    return result


def daysInMonths(month, year):
    if isLeapYear(year) and month == 2:
        return 29
    else:
        return daysofMonths[month - 1]


def nextDay(year, month, day):
    if day < daysInMonths(month, year):
        return year, month, day + 1
    else:
        if month == 12:
            return year + 1, 1, 1
        else:
            return year, month + 1, 1


def dateIsBefore(year1, month1, day1, year2, month2, day2):
    """Returns True if year1-month1-day1 is before year2-month2-day2. Otherwise, returns False."""
    if year1 < year2:
        return True
    if year1 == year2:
        if month1 < month2:
            return True
        if month1 == month2:
            return day1 < day2
    return False


def daysBetweenDates(year1, month1, day1, year2, month2, day2):
    """Returns the number of days between year1/month1/day1
       and year2/month2/day2. Assumes inputs are valid dates
       in Gregorian calendar."""

    # program defensively! Add an assertion if the input is not valid!
    assert not dateIsBefore(year2, month2, day2, year1, month1, day1)

    days = 0
    while dateIsBefore(year1, month1, day1, year2, month2, day2):
        year1, month1, day1 = nextDay(year1, month1, day1)
        days += 1
    return days


def test():
    assert daysBetweenDates(2013, 1, 1, 2013, 1, 1) == 0
    assert daysBetweenDates(2013, 1, 1, 2013, 1, 2) == 1
    assert daysBetweenDates(2013, 1, 1, 2014, 1, 1) == 365
    assert isLeapYear(2001) is False
    assert daysInMonths(2, 2001) == 28
    assert daysInMonths(2, 2000) == 29
    assert nextDay(2013, 1, 1) == (2013, 1, 2)
    assert nextDay(2012, 2, 28) == (2012, 2, 29)
    assert nextDay(2013, 12, 31) == (2014, 1, 1)
    assert nextDay(2001, 2, 28) == (2001, 3, 1)
    print('tese finished.')


test()
