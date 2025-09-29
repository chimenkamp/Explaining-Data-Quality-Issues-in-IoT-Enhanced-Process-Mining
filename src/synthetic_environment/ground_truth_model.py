"""Ground truth Petri net model generator"""
from typing import Dict, List, Tuple, Any
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils


def create_manufacturing_ground_truth() -> Tuple[PetriNet, Marking, Marking]:
    """
    Create ground truth Petri net for manufacturing process.

    :return: Tuple of (petri_net, initial_marking, final_marking)
    """
    net = PetriNet("manufacturing_process")

    source = PetriNet.Place("source")
    sink = PetriNet.Place("sink")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    p4 = PetriNet.Place("p4")
    p5 = PetriNet.Place("p5")

    net.places.add(source)
    net.places.add(sink)
    net.places.add(p1)
    net.places.add(p2)
    net.places.add(p3)
    net.places.add(p4)
    net.places.add(p5)

    weld_position = PetriNet.Transition("Welding_Position", "Welding_Position")
    weld = PetriNet.Transition("Weld", "Weld")
    cool = PetriNet.Transition("Cool", "Cool")
    inspect_position = PetriNet.Transition("Inspection_Position", "Inspection_Position")
    scan = PetriNet.Transition("Scan", "Scan")
    measure = PetriNet.Transition("Measure", "Measure")
    validate = PetriNet.Transition("Validate", "Validate")
    package_pick = PetriNet.Transition("Package_Pick", "Package_Pick")
    package = PetriNet.Transition("Package", "Package")
    seal = PetriNet.Transition("Seal", "Seal")

    net.transitions.add(weld_position)
    net.transitions.add(weld)
    net.transitions.add(cool)
    net.transitions.add(inspect_position)
    net.transitions.add(scan)
    net.transitions.add(measure)
    net.transitions.add(validate)
    net.transitions.add(package_pick)
    net.transitions.add(package)
    net.transitions.add(seal)

    petri_utils.add_arc_from_to(source, weld_position, net)
    petri_utils.add_arc_from_to(weld_position, p1, net)
    petri_utils.add_arc_from_to(p1, weld, net)
    petri_utils.add_arc_from_to(weld, p2, net)
    petri_utils.add_arc_from_to(p2, cool, net)
    petri_utils.add_arc_from_to(cool, p3, net)
    petri_utils.add_arc_from_to(p3, inspect_position, net)
    petri_utils.add_arc_from_to(inspect_position, p4, net)
    petri_utils.add_arc_from_to(p4, scan, net)
    petri_utils.add_arc_from_to(scan, p4, net)
    petri_utils.add_arc_from_to(p4, measure, net)
    petri_utils.add_arc_from_to(measure, p4, net)
    petri_utils.add_arc_from_to(p4, validate, net)
    petri_utils.add_arc_from_to(validate, p5, net)
    petri_utils.add_arc_from_to(p5, package_pick, net)
    petri_utils.add_arc_from_to(package_pick, p5, net)
    petri_utils.add_arc_from_to(p5, package, net)
    petri_utils.add_arc_from_to(package, p5, net)
    petri_utils.add_arc_from_to(p5, seal, net)
    petri_utils.add_arc_from_to(seal, sink, net)

    initial_marking = Marking()
    initial_marking[source] = 1

    final_marking = Marking()
    final_marking[sink] = 1

    return net, initial_marking, final_marking


def get_activity_sensor_mapping() -> Dict[str, List[str]]:
    """
    Get mapping from activities to sensor types.

    :return: Dictionary mapping activity names to sensor type list
    """
    return {
        'Welding_Position': ['PWR', 'POS'],
        'Weld': ['PWR', 'TEMP'],
        'Cool': ['TEMP'],
        'Inspection_Position': ['POS', 'VIB'],
        'Scan': ['VIB', 'POS'],
        'Measure': ['VIB', 'POS'],
        'Validate': ['VIB'],
        'Package_Pick': ['POS', 'PWR'],
        'Package': ['PWR', 'POS'],
        'Seal': ['PWR']
    }


def get_activity_duration_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get typical duration ranges for each activity in seconds.

    :return: Dictionary mapping activity names to (min_duration, max_duration) tuples
    """
    return {
        'Welding_Position': (3.0, 6.0),
        'Weld': (8.0, 15.0),
        'Cool': (10.0, 20.0),
        'Inspection_Position': (2.0, 5.0),
        'Scan': (4.0, 8.0),
        'Measure': (5.0, 10.0),
        'Validate': (3.0, 6.0),
        'Package_Pick': (2.0, 4.0),
        'Package': (5.0, 10.0),
        'Seal': (6.0, 12.0)
    }