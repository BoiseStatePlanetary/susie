class TransitTimes(object):
    """Docstrings for transit times object.
    """
    def __init__(self, epochs, mid_transit_times, uncertainties=None):
        self.epochs = epochs
        self.mid_transit_times = mid_transit_times
        if uncertainties == None:
            # make an array of 1s in the same shape of epochs and mid_transit_times
            pass
        else:
            self.uncertainties = uncertainties
    