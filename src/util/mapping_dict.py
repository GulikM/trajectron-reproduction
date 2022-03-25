class MappingDict(dict):
    """
    Class for use as config mapping: returns key if self[key] does not exist.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        """
        Method called when an item is not found in dictionary.
        :param key: key that is missing
        :return: value of key
        """
        return key

    def __getitem__(self, item):
        """
        Overriding __getitem__ to type-convert nested dictionaries.
        :param item: item to get
        :return: self[item]
        """
        result = super().__getitem__(item)
        if isinstance(result, dict) and not isinstance(result, type(self)):
            self.__setitem__(item, MappingDict(result))
            return super().__getitem__(item)
        return result
