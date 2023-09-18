import numpy as np
from model_ephemeris import ModelEphemeris
from transit_times import TransitTimes

class Ephemeris(object):
    """Docstring about the ephemeris object.
    """
    def __init__(self, transit_times, model_ephemerides=None):
        # initializing the transit times object and model ephermeris object
        # NOTE: We will have the user create their own transit times obj and send that to the ephemeris obj
        # NOTE: Do we want there to be a list of model ephemerides or just hold one at a time?
        self.model_ephemerides = []
        self.transit_times = transit_times
        if model_ephemerides is not None:
            self.model_ephemeris = model_ephemerides

    def get_model_parameters(self, model_type, kwargs):
        # NOTE: Do we want to have the user control their x, y, yerr inputs or do we want to pull them straight from the transit times obj?
        # Step 1: Get data from transit times obj
        x = self.transit_times.epochs - np.min(self.transit_times.epochs)
        y = self.transit_times.mid_transit_times - np.min(self.transit_times.mid_transit_times)
        yerr = self.transit_times.uncertainties
        # Step 2: Instantiate model ephemeris object
        model_ephemeris = ModelEphemeris()
        # Step 3: Call get_model_parameters to run model and get back parameters
        return_data = model_ephemeris.get_model_parameters(model_type, x, y, yerr, kwargs) # this will return the model parameters as dict
        # Step 4: Append this new model to the list of models
        self.model_ephemerides.append(model_ephemeris)
        # Step 5: Return the model parameter data to the user so they can see what's going on 
        return return_data



"""
example:

ephemeris = Ephemeris()

I have a CSV file with 3 columns: epochs, transit times, and uncertainties
I have to split them into arrays and then add them one by one to the transit times object

"""