import logging

class MultilineFormatter(logging.Formatter):
    def format(self, record):
        string = logging.Formatter.format(self, record)
        header, footer = string.split(record.message)
        string = string.replace('\n', '\n' + ' '*len(header))
        return string

