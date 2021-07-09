import os
import datetime

from reoccuring_drift_stream import ReoccuringDriftStream 
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
#Abrupt Concept Drift Generators
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.sine_generator import SineGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
# Incremental Concept Drift Generators
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
# No Concept Drift Generators
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator

class Study():
    """ Study

    Abstract class for evaluation methods. Initializes streams for studies and is baseline class for metrics, paths and dates.   

    Parameters
    ----------
    streams: list(Stream)
        List of streams which will be evaluated. If no streams given, standard streams are initialized.

    path: String (Default: /)
        Path to directory for save of study results. Default in current directory.
        Folder will be created if not existent. 
        
    Notes
    -----
    This is a abstrace base class. No studies should be processed. 
   
    """
   
    # TODO: List of string with stream names for individual studies
    def __init__(self, streams=None, path="/"):
        if streams == None:
                self.streams = self.init_standard_streams()
        else:
            self.streams = streams
        self.path = path
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.date_format = "%Y-%m-%d %H-%M"
        self.metrics = ['accuracy', 'kappa_t', 'kappa_m', 'kappa']
        self.date = str(datetime.datetime.now().strftime(self.date_format))
        self.chwd_root()
        try:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
        except Exception as e: raise FileNotFoundError("Error while creating Directory!")
    
    def init_standard_streams(self):
        """Initialize standard data streams
        
        Standard streams are inspired by the experiment settings of 
        Gomes, Heitor Murilo & Bifet, Albert & Read, Jesse & Barddal, Jean Paul & 
        Enembreck, Fabrício & Pfahringer, Bernhard & Holmes, Geoff & 
        Abdessalem, Talel. (2017). Adaptive random forests for evolving data 
        stream classification. Machine Learning. 1-27. 10.1007/s10994-017-5642-8. 
        """
        mixed1 = MIXEDGenerator(classification_function=0, random_state=112, balance_classes=False)
        mixed2 = MIXEDGenerator(classification_function=1, random_state=112, balance_classes=False)
        ra_mixed = ConceptDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,width=1)
        rg_mixed = ConceptDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,width=1000)
        rg_mixed.name = "mixed_g"
        ra_mixed.name = "mixed_a"


        hyper = HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1)
        
        led_a = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            alpha=90.0, # angle of change grade 0 - 90
                            position=250000,
                            width=1)
 
        led_a.name = "led_a"
        led_g = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            position=250000,
                            width=50000)
        led_g.name = "led_g"
        rand_tree = RandomTreeGenerator()
        rand_tree.name = "rand_tree" 
        rbf_if = RandomRBFGeneratorDrift(change_speed=0.001)
        rbf_if.name = "rbf_if"
        rbf_im = RandomRBFGeneratorDrift(change_speed=0.0001)
        rbf_im.name = "rbf_im"
        sea_a = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                            drift_stream=SEAGenerator(random_state=112, 
                                                          classification_function=2, noise_percentage=0.1),
                            alpha=90.0,
                            random_state=None,
                            position=250000,
                            width=1)  
        sea_a.name = "sea_a"                            
        sea_g = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                            drift_stream=SEAGenerator(random_state=112, 
                                                          classification_function=1, noise_percentage=0.1),
                            random_state=None,
                            position=250000,
                            width=50000)
        sea_g.name = "sea_g"                  
        return [ra_mixed, rg_mixed, hyper, rand_tree, rbf_if, rbf_im, sea_a, sea_g]
        
    def init_reoccuring_streams(self):
        """Initialize reoccuring data streams: abrupt and gradual"""
        s1 = SineGenerator(classification_function=0, balance_classes=False, random_state=112)
        s2 = SineGenerator(classification_function=1, balance_classes=False, random_state=112)
        ra_sine = ReoccuringDriftStream(stream=s1, drift_stream=s2, alpha=90.0, position=2000, width=1)
        rg_sine = ReoccuringDriftStream(stream=s1, drift_stream=s2, alpha=90.0, position=2000, width=1000)

        stagger1 = STAGGERGenerator(classification_function=0, balance_classes=False, random_state=112)
        stagger2 = STAGGERGenerator(classification_function=1, balance_classes=False, random_state=112)
        ra_stagger = ReoccuringDriftStream(stream=stagger1, drift_stream=stagger2, random_state=112, alpha=90.0,position=2000,width=1)
        rg_stagger = ReoccuringDriftStream(stream=stagger1, drift_stream=stagger2, random_state=112, alpha=90.0,position=2000,width=1000)

        sea1 = SEAGenerator(classification_function=0, balance_classes=False, random_state=112)
        sea2 = SEAGenerator(classification_function=1, balance_classes=False, random_state=112)
        ra_sea = ReoccuringDriftStream(stream=sea1, drift_stream=sea2, random_state=112, alpha=90.0, position=2000,width=1)
        rg_sea = ReoccuringDriftStream(stream=sea1, drift_stream=sea2, random_state=112, alpha=90.0, position=2000,width=1000)

        mixed1 = MIXEDGenerator(classification_function=0, random_state=112, balance_classes=False)
        mixed2 = MIXEDGenerator(classification_function=1, random_state=112, balance_classes=False)
        ra_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,width=1)
        rg_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,width=1000)

        inc = HyperplaneGenerator(random_state=112)
        
        return [ra_sine, rg_sine, ra_stagger, rg_stagger, ra_sea, rg_sea, ra_mixed, rg_mixed, inc]
    
    def init_reoccuring_standard_streams(self):
        """Initialize the standard streams as reoccuring
        We can only introduce reoccuring drift on generators where abrupt or gradual drift is possible.
        This means all standard streams except RBF, HYPER and RTG"""

        mixed1 = MIXEDGenerator(classification_function=0, random_state=112, balance_classes=False)
        mixed2 = MIXEDGenerator(classification_function=1, random_state=112, balance_classes=False)
        ra_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,width=1)
        rg_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,width=1000)


        led_a = ReoccuringDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            alpha=90.0, # angle of change grade 0 - 90
                            position=2000,
                            width=1)
 

        led_g = ReoccuringDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            position=2000,
                            width=1000)

        sea_a = ReoccuringDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                            drift_stream=SEAGenerator(random_state=112, 
                                                          classification_function=2, noise_percentage=0.1),
                            alpha=90.0,
                            random_state=None,
                            position=250000,
                            width=1)  

        sea_g = ReoccuringDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                            drift_stream=SEAGenerator(random_state=112, 
                                                          classification_function=1, noise_percentage=0.1),
                            random_state=None,
                            position=250000,
                            width=1000)

            
        return [ra_mixed, rg_mixed, sea_a, sea_g]

    def init_esann_si_streams(self):
        mixed1 = MIXEDGenerator(classification_function=0, random_state=112, balance_classes=False)
        mixed2 = MIXEDGenerator(classification_function=1, random_state=112, balance_classes=False)
        ra_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0,
                                         position=2000, width=1)
        rg_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0,
                                         position=2000, width=1000)


        sea_ra = ReoccuringDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1),
                                      drift_stream=SEAGenerator(random_state=112,
                                                                classification_function=2, noise_percentage=0.1),
                                      alpha=90.0,
                                      random_state=None,
                                      position=250000,
                                      width=1)

        sea_rg = ReoccuringDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1),
                                      drift_stream=SEAGenerator(random_state=112,
                                                                classification_function=1, noise_percentage=0.1),
                                      random_state=None,
                                      position=250000,
                                      width=1000)
        mixed1 = MIXEDGenerator(classification_function=0, random_state=112, balance_classes=False)
        mixed2 = MIXEDGenerator(classification_function=1, random_state=112, balance_classes=False)
        a_mixed = ConceptDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,
                                      width=1)
        g_mixed = ConceptDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112, alpha=90.0, position=2000,
                                      width=1000)
        a_mixed.name = "mixed_g"
        g_mixed.name = "mixed_a"

        hyper = HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1)
        rand_tree = RandomTreeGenerator()
        rand_tree.name = "rand_tree"
        rbf_if = RandomRBFGeneratorDrift(change_speed=0.001)
        rbf_if.name = "rbf_if"
        rbf_im = RandomRBFGeneratorDrift(change_speed=0.0001)
        rbf_im.name = "rbf_im"
        sea_a = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1),
                                   drift_stream=SEAGenerator(random_state=112,
                                                             classification_function=2, noise_percentage=0.1),
                                   alpha=90.0,
                                   random_state=None,
                                   position=250000,
                                   width=1)
        sea_a.name = "sea_a"
        sea_g = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1),
                                   drift_stream=SEAGenerator(random_state=112,
                                                             classification_function=1, noise_percentage=0.1),
                                   random_state=None,
                                   position=250000,
                                   width=50000)
        sea_g.name = "sea_g"

        """Initialize real world data streams, will be loaded from file"""

        if not os.path.join("..", "..", "datasets"):
            raise FileNotFoundError("Folder for data cannot be found! Should be datasets")
        os.chdir(os.path.join("..", "..", "datasets"))
        try:
            covertype = FileStream(os.path.realpath('covtype.csv'))  # Label failure
            covertype.name = "covertype"
            elec = FileStream(os.path.realpath('elec.csv'))
            elec.name = "elec"
            poker = FileStream(os.path.realpath('poker.csv'))  # label failure
            poker.name = "poker"
            weather = FileStream(os.path.realpath('weather.csv'))
            weather.name = "weather"
            gmsc = FileStream(os.path.realpath('gmsc.csv'))
            gmsc.name = "gmsc"
            #  airlines = FileStream(os.path.realpath('airlines.csv')) #label failure
            moving_squares = FileStream(os.path.realpath('moving_squares.csv'))
            moving_squares.name = "moving_squares"

        except Exception as e:
            raise FileNotFoundError("Real-world datasets can't loaded! Check directory ")
        return [covertype, elec, poker, weather, gmsc, moving_squares,a_mixed, g_mixed, hyper, rand_tree, rbf_if, rbf_im, sea_a, sea_g,ra_mixed, rg_mixed,  sea_ra, sea_rg,]

    def init_real_world(self):
        """Initialize real world data streams, will be loaded from file"""
    
        if not os.path.join("..","..","datasets"):
            raise FileNotFoundError("Folder for data cannot be found! Should be datasets")
        os.chdir(os.path.join("..","..","datasets"))
        try:   
            covertype = FileStream(os.path.realpath('covtype.csv')) # Label failure
            covertype.name = "covertype"
            elec = FileStream(os.path.realpath('elec.csv'))
            elec.name = "elec"
            poker = FileStream(os.path.realpath('poker.csv')) #label failure
            poker.name = "poker"
            weather = FileStream(os.path.realpath('weather.csv'))
            weather.name = "weather"
            gmsc = FileStream(os.path.realpath('gmsc.csv'))
            gmsc.name = "gmsc"
          #  airlines = FileStream(os.path.realpath('airlines.csv')) #label failure
            moving_squares = FileStream(os.path.realpath('moving_squares.csv'))
            moving_squares.name = "moving_squares"
            return [covertype,elec, poker, weather, gmsc, moving_squares]
        except Exception as e: 
            raise FileNotFoundError("Real-world datasets can't loaded! Check directory ")
            return []
        
    def chwd_root(self):
        os.chdir(self.root_dir)
