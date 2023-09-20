import numpy as np
from model_ephemeris import ModelEphemeris
from transit_times import TransitTimes

class Ephemeris(object):
    # TODO: Merge model ephemeris into this obj
    """Docstring about the ephemeris object.
    """
    def __init__(self, transit_times, model_ephemerides=None):
        # initializing the transit times object and model ephermeris object
        # NOTE: We will have the user create their own transit times obj and send that to the ephemeris obj
        # NOTE: Do we want there to be a list of model ephemerides or just hold one at a time?
        self.model_ephemerides = model_ephemerides
        self.transit_times = transit_times
        if model_ephemerides is None:
            self.model_ephemeris = []
        self._validate()

    def _validate(self):
        # Check that transit_times is an instance of the TransitTimes object
        if not isinstance(self.transit_times, TransitTimes):
            raise ValueError("Variable 'transit_times' expected type of object 'TransitTimes'.")
        # Check that if model_ephemerides is given, it is a list of ModelEphemeris objects **will do if we keep it that way
        
    def get_model_parameters(self, model_type, **kwargs):
        # NOTE: Do we want to have the user be able to store the model, do we want to store it for them (if so, as a dict or array)
        # NOTE: Do we want to return the model ephemeris object to the user or just the return data dict? *Would help to keep in mind how users may use this going further into package use
        # Step 1: Get data from transit times obj
        x = self.transit_times.epochs - np.min(self.transit_times.epochs)
        y = self.transit_times.mid_transit_times - np.min(self.transit_times.mid_transit_times)
        yerr = self.transit_times.uncertainties
        # Step 2: Instantiate model ephemeris object
        model_ephemeris = ModelEphemeris()
        # Step 3: Call get_model_parameters to run model and get back parameters
        model_ephemeris.get_model_parameters(model_type, x, y, yerr, **kwargs) # this will return the model parameters as dict
        # Step 4: Append this new model to the list of models
        self.model_ephemerides.append(model_ephemeris)
        # Step 5: Return the model ephemeris to the user so they can handle it
        return model_ephemeris
    


if __name__ == '__main__':
    # This is for small tests, will only run if file is called directly
    data = np.genfromtxt("./malia_examples/WASP12b_transit_ephemeris.csv", delimiter=',', names=True)
    epochs = data['epoch'].astype('int')
    test_tt = TransitTimes(epochs, data['transit_time'], data['sigma_transit_time'])
    test = Ephemeris(test_tt)
    print(vars(test))

    mp = test.get_model_parameters('quadratic')
    print(vars(mp))
    print(vars(test))
