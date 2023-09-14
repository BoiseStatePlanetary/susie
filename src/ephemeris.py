from model_ephemeris import ModelEphemeris
from transit_times import TransitTimes

class Ephemeris(object):
    """Docstring about the ephemeris object.
    """
    def __init__(self, epochs, mid_transit_times, uncertainties=None, transit_times=None, model_ephemeris=None):
        # initializing the transit times object and model ephermeris object
        # NOTE: Do we want to have the user start off with giving us the transit data or do we want them to build their own transit time object and add that into the ephemeris object themselves?
        self.model_ephemeris = ModelEphemeris()
        self.transit_times = TransitTimes(epochs, mid_transit_times, uncertainties)
        if model_ephemeris is not None:
            self.model_ephemeris = model_ephemeris
        if transit_times is not None:
            self.transit_times = transit_times



"""
example:

ephemeris = Ephemeris()

I have a CSV file with 3 columns: epochs, transit times, and uncertainties
I have to split them into arrays and then add them one by one to the transit times object

ephemeris.transit_times.epochs = 

"""