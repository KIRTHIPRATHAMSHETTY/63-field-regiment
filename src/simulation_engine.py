import numpy as np
from typing import Dict, Any


def simulate_scenario(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate a military scenario and return outcome metrics.
    
    Args:
        config: Dictionary containing:
            - side_a_force_strength, side_b_force_strength (0-100)
            - side_a_mobility, side_b_mobility (0-1)
            - side_a_logistics, side_b_logistics (0-1)
            - side_a_intelligence, side_b_intelligence (0-1)
            - side_a_communication, side_b_communication (0-1)
            - side_a_readiness, side_b_readiness (0-1)
            - side_a_morale, side_b_morale (0.0-1.0)
            - side_a_training, side_b_training (0.0-1.0)
            - side_a_supply_distance, side_b_supply_distance (0-500 km)
            - side_a_surprise, side_b_surprise (0.0-1.0)
            - side_a_fatigue, side_b_fatigue (0.0-1.0): 0=fresh, 1=exhausted
            - side_a_equipment_quality, side_b_equipment_quality (0.0-1.0)
            - electronic_warfare_level (0.0-1.0)
            - objective_type (str): Attack, Defend, Delay, Withdraw
            - season (str): Spring, Summer, Autumn, Winter
            - rules_of_engagement (str): Permissive, Restrictive, Defensive, Offensive
            - terrain_type (str): Open, Hilly, Forest, Desert, Urban, Mixed
            - weather (str): Clear, Cloudy, Rain, Fog, Extreme
            - visibility (str): High, Medium, Low
            - time_of_day (str): Day, Night
            - environmental_difficulty (float): 0-1
    
    Returns:
        Dictionary with outcome, duration, scores, and resource status
    """
    # Extract parameters
    force_a = config['side_a_force_strength']
    force_b = config['side_b_force_strength']
    mobility_a = config['side_a_mobility']
    mobility_b = config['side_b_mobility']
    logistics_a = config['side_a_logistics']
    logistics_b = config['side_b_logistics']
    intel_a = config['side_a_intelligence']
    intel_b = config['side_b_intelligence']
    comm_a = config['side_a_communication']
    comm_b = config['side_b_communication']
    readiness_a = config['side_a_readiness']
    readiness_b = config['side_b_readiness']
    
    # New extended parameters
    morale_a = config.get('side_a_morale', 0.7)
    morale_b = config.get('side_b_morale', 0.7)
    training_a = config.get('side_a_training', 0.7)
    training_b = config.get('side_b_training', 0.7)
    supply_dist_a = config.get('side_a_supply_distance', 100)
    supply_dist_b = config.get('side_b_supply_distance', 100)
    surprise_a = config.get('side_a_surprise', 0.3)
    surprise_b = config.get('side_b_surprise', 0.3)
    ew_level = config.get('electronic_warfare_level', 0.0)
    objective_type = config.get('objective_type', 'Attack')

    # New v2 parameters
    fatigue_a = config.get('side_a_fatigue', 0.3)
    fatigue_b = config.get('side_b_fatigue', 0.3)
    equip_a = config.get('side_a_equipment_quality', 0.7)
    equip_b = config.get('side_b_equipment_quality', 0.7)
    season = config.get('season', 'Summer')
    roe = config.get('rules_of_engagement', 'Permissive')

    terrain = config['terrain_type']
    weather = config['weather']
    visibility = config['visibility']
    time_of_day = config['time_of_day']
    env_difficulty = config['environmental_difficulty']

    
    # Normalize force to 0-1
    force_a_norm = force_a / 100.0
    force_b_norm = force_b / 100.0
    
    # Apply terrain modifiers
    mobility_mod_a, mobility_mod_b = 1.0, 1.0
    readiness_mod_a, readiness_mod_b = 1.0, 1.0
    intel_mod_a, intel_mod_b = 1.0, 1.0
    logistics_mod_a, logistics_mod_b = 1.0, 1.0
    comm_mod_a, comm_mod_b = 1.0, 1.0
    
    if terrain == 'Open':
        mobility_mod_a = mobility_mod_b = 1.3
        readiness_mod_a = readiness_mod_b = 0.9
    elif terrain == 'Hilly':
        readiness_mod_a = readiness_mod_b = 1.3
        mobility_mod_a = mobility_mod_b = 0.8
    elif terrain == 'Forest':
        intel_mod_a = intel_mod_b = 1.2
        mobility_mod_a = mobility_mod_b = 0.85
    elif terrain == 'Desert':
        logistics_mod_a = logistics_mod_b = 1.3
        mobility_mod_a = mobility_mod_b = 0.9
    elif terrain == 'Urban':
        readiness_mod_a = readiness_mod_b = 1.15
        comm_mod_a = comm_mod_b = 1.15
    elif terrain == 'Mixed':
        pass  # All modifiers remain 1.0
    
    # Apply weather modifiers
    if weather == 'Clear':
        pass  # All ×1.0
    elif weather == 'Cloudy':
        global_weather_mod = 0.95
        mobility_mod_a *= global_weather_mod
        mobility_mod_b *= global_weather_mod
        readiness_mod_a *= global_weather_mod
        readiness_mod_b *= global_weather_mod
        intel_mod_a *= global_weather_mod
        intel_mod_b *= global_weather_mod
        logistics_mod_a *= global_weather_mod
        logistics_mod_b *= global_weather_mod
        comm_mod_a *= global_weather_mod
        comm_mod_b *= global_weather_mod
    elif weather == 'Rain':
        logistics_mod_a *= 0.85
        logistics_mod_b *= 0.85
        mobility_mod_a *= 0.85
        mobility_mod_b *= 0.85
    elif weather == 'Fog':
        # Intel ×1.25 for higher side, ×0.75 for lower side
        if intel_a > intel_b:
            intel_mod_a *= 1.25
            intel_mod_b *= 0.75
        else:
            intel_mod_b *= 1.25
            intel_mod_a *= 0.75
    elif weather == 'Extreme':
        global_weather_mod = 0.75
        mobility_mod_a *= global_weather_mod
        mobility_mod_b *= global_weather_mod
        readiness_mod_a *= global_weather_mod
        readiness_mod_b *= global_weather_mod
        intel_mod_a *= global_weather_mod
        intel_mod_b *= global_weather_mod
        logistics_mod_a *= global_weather_mod
        logistics_mod_b *= global_weather_mod
        comm_mod_a *= global_weather_mod
        comm_mod_b *= global_weather_mod
    
    # Apply visibility modifier (affects intelligence)
    if visibility == 'High':
        pass  # ×1.0
    elif visibility == 'Medium':
        intel_mod_a *= 0.95
        intel_mod_b *= 0.95
    elif visibility == 'Low':
        intel_mod_a *= 0.85
        intel_mod_b *= 0.85
    
    # Apply time of day modifier
    if time_of_day == 'Day':
        pass  # No modifier
    elif time_of_day == 'Night':
        intel_mod_a *= 1.15
        intel_mod_b *= 1.15
        comm_mod_a *= 0.9
        comm_mod_b *= 0.9
    
    # Apply season modifiers (affect logistics, mobility, intel)
    if season == 'Spring':
        logistics_mod_a *= 0.95
        logistics_mod_b *= 0.95
    elif season == 'Summer':
        pass  # Neutral season
    elif season == 'Autumn':
        mobility_mod_a *= 0.92
        mobility_mod_b *= 0.92
    elif season == 'Winter':
        mobility_mod_a *= 0.80
        mobility_mod_b *= 0.80
        logistics_mod_a *= 0.85
        logistics_mod_b *= 0.85
        intel_mod_a *= 0.90
        intel_mod_b *= 0.90

    # Apply Rules of Engagement modifiers
    # These shape how aggressively each side can exploit advantages
    roe_comm_mod = 1.0
    roe_score_bonus_a = 0.0
    roe_score_bonus_b = 0.0
    if roe == 'Permissive':
        # Full freedom of action — both sides fight at full capacity
        pass
    elif roe == 'Restrictive':
        # Constrained action — reduced effectiveness for both
        roe_comm_mod = 0.85
        comm_mod_a *= roe_comm_mod
        comm_mod_b *= roe_comm_mod
    elif roe == 'Defensive':
        # Side B gains defensive bonus; Side A (attacker) penalized
        roe_score_bonus_b = 0.07
        roe_score_bonus_a = -0.04
    elif roe == 'Offensive':
        # Side A (attacker) gains offensive bonus; Side B penalized
        roe_score_bonus_a = 0.07
        roe_score_bonus_b = -0.04

    # Apply environmental difficulty penalty
    env_penalty = 0.1 * env_difficulty

    
    # Apply extended scenario modifiers
    
    # 1. Morale: Multiply readiness contribution by (0.7 + 0.3 * morale)
    morale_mod_a = 0.7 + 0.3 * morale_a
    morale_mod_b = 0.7 + 0.3 * morale_b
    readiness_mod_a *= morale_mod_a
    readiness_mod_b *= morale_mod_b
    
    # 2. Training: Multiply force + intelligence contributions by (0.8 + 0.2 * training)
    training_mod_a = 0.8 + 0.2 * training_a
    training_mod_b = 0.8 + 0.2 * training_b
    
    # 3. Supply Distance: Multiply logistics by max(0.5, 1 - distance/1000)
    supply_mod_a = max(0.5, 1 - supply_dist_a / 1000.0)
    supply_mod_b = max(0.5, 1 - supply_dist_b / 1000.0)
    logistics_mod_a *= supply_mod_a
    logistics_mod_b *= supply_mod_b
    
    # 4. Electronic Warfare: effective_communication = communication * (1 - ew_level * 0.6)
    ew_mod = 1 - ew_level * 0.6
    comm_mod_a *= ew_mod
    comm_mod_b *= ew_mod

    # 5. Fatigue: higher fatigue reduces readiness and mobility
    #    fatigue 0.0 = fresh (full capacity), 1.0 = exhausted (−30% readiness, −20% mobility)
    fatigue_readiness_mod_a = max(0.5, 1.0 - 0.3 * fatigue_a)
    fatigue_readiness_mod_b = max(0.5, 1.0 - 0.3 * fatigue_b)
    fatigue_mobility_mod_a = max(0.6, 1.0 - 0.2 * fatigue_a)
    fatigue_mobility_mod_b = max(0.6, 1.0 - 0.2 * fatigue_b)
    readiness_mod_a *= fatigue_readiness_mod_a
    readiness_mod_b *= fatigue_readiness_mod_b
    mobility_mod_a *= fatigue_mobility_mod_a
    mobility_mod_b *= fatigue_mobility_mod_b

    # 6. Equipment Quality: scales force strength and mobility effectiveness
    #    0.0 = poor equipment (0.75× force/mobility), 1.0 = elite gear (1.15× force/mobility)
    equip_mod_a = 0.75 + 0.40 * equip_a
    equip_mod_b = 0.75 + 0.40 * equip_b
    
    # Calculate effective values with modifiers
    mobility_eff_a = max(0, min(1, (mobility_a * mobility_mod_a * equip_mod_a) - env_penalty))
    mobility_eff_b = max(0, min(1, (mobility_b * mobility_mod_b * equip_mod_b) - env_penalty))
    
    intel_eff_a = max(0, min(1, (intel_a * intel_mod_a) - env_penalty))
    intel_eff_b = max(0, min(1, (intel_b * intel_mod_b) - env_penalty))
    
    logistics_eff_a = max(0, min(1, (logistics_a * logistics_mod_a) - env_penalty))
    logistics_eff_b = max(0, min(1, (logistics_b * logistics_mod_b) - env_penalty))
    
    comm_eff_a = max(0, min(1, (comm_a * comm_mod_a) - env_penalty))
    comm_eff_b = max(0, min(1, (comm_b * comm_mod_b) - env_penalty))
    
    readiness_eff_a = max(0, min(1, (readiness_a * readiness_mod_a) - env_penalty))
    readiness_eff_b = max(0, min(1, (readiness_b * readiness_mod_b) - env_penalty))
    
    # Base score calculation
    # Base: 0.25*force + 0.15*mobility + 0.15*logistics + 0.15*intelligence + 0.10*communication + 0.20*readiness
    # Apply training modifier to force and intelligence; equipment modifier to force
    score_a = (
        0.25 * force_a_norm * training_mod_a * equip_mod_a +
        0.15 * mobility_eff_a +
        0.15 * logistics_eff_a +
        0.15 * intel_eff_a * training_mod_a +
        0.10 * comm_eff_a +
        0.20 * readiness_eff_a
    )
    
    score_b = (
        0.25 * force_b_norm * training_mod_b * equip_mod_b +
        0.15 * mobility_eff_b +
        0.15 * logistics_eff_b +
        0.15 * intel_eff_b * training_mod_b +
        0.10 * comm_eff_b +
        0.20 * readiness_eff_b
    )
    
    # Interaction terms
    mobility_effective = (mobility_eff_a - mobility_eff_b) * 0.1
    intel_effective = (intel_eff_a - intel_eff_b) * 0.1
    
    # Final scores with interaction terms and Gaussian noise
    noise = np.random.normal(0, 0.015)
    
    simulation_score_a = score_a + mobility_effective + intel_effective + noise
    simulation_score_b = score_b - mobility_effective - intel_effective + noise
    
    # Surprise: If surprise > 0.7, add +0.15 to that side's final score
    if surprise_a > 0.7:
        simulation_score_a += 0.15
    if surprise_b > 0.7:
        simulation_score_b += 0.15

    # Rules of Engagement score bonuses
    simulation_score_a += roe_score_bonus_a
    simulation_score_b += roe_score_bonus_b

    # 5. Objective Type modifiers
    decisive_threshold = 0.15  # Default threshold for decisive outcome
    
    if objective_type == 'Attack':
        # +0.05 to side A score, requires +0.10 logistics
        simulation_score_a += 0.05
        if logistics_a < 0.6:  # Penalty if logistics insufficient
            simulation_score_a -= 0.03
    elif objective_type == 'Defend':
        # +0.10 to side B score if terrain is Hilly/Forest/Urban
        if terrain in ['Hilly', 'Forest', 'Urban']:
            simulation_score_b += 0.10
    elif objective_type == 'Delay':
        # Reduces decisive thresholds (more stalemates)
        decisive_threshold = 0.10
    elif objective_type == 'Withdraw':
        # Reduces stalemate probability
        decisive_threshold = 0.20
    
    # Determine outcome
    diff = simulation_score_a - simulation_score_b
    
    if diff > decisive_threshold:
        outcome = 'SIDE_A_ADVANTAGE'
    elif diff < -decisive_threshold:
        outcome = 'SIDE_B_ADVANTAGE'
    else:
        outcome = 'STALEMATE'
    
    # Calculate duration (in days) based on score difference
    # Closer scores = longer duration
    base_duration = 2.0 + (1.0 - abs(diff)) * 3.0
    duration = max(0.5, min(7.0, base_duration + np.random.normal(0, 0.3)))
    
    # Calculate resource status
    # Higher score = better resource status
    resource_status_a = 'High' if simulation_score_a > 0.6 else ('Medium' if simulation_score_a > 0.4 else 'Low')
    resource_status_b = 'High' if simulation_score_b > 0.6 else ('Medium' if simulation_score_b > 0.4 else 'Low')
    
    return {
        'outcome': outcome,
        'duration': round(duration, 1),
        'simulation_score_a': round(simulation_score_a, 3),
        'simulation_score_b': round(simulation_score_b, 3),
        'resource_status_a': resource_status_a,
        'resource_status_b': resource_status_b
    }


if __name__ == '__main__':
    # Test the simulation engine
    test_config = {
        'side_a_force_strength': 80,
        'side_b_force_strength': 70,
        'side_a_mobility': 0.75,
        'side_b_mobility': 0.60,
        'side_a_logistics': 0.70,
        'side_b_logistics': 0.55,
        'side_a_intelligence': 0.80,
        'side_b_intelligence': 0.60,
        'side_a_communication': 0.85,
        'side_b_communication': 0.65,
        'side_a_readiness': 0.75,
        'side_b_readiness': 0.60,
        'side_a_morale': 0.8,
        'side_b_morale': 0.6,
        'side_a_training': 0.75,
        'side_b_training': 0.55,
        'side_a_supply_distance': 80,
        'side_b_supply_distance': 150,
        'side_a_surprise': 0.6,
        'side_b_surprise': 0.3,
        'electronic_warfare_level': 0.25,
        'objective_type': 'Attack',
        'terrain_type': 'Hilly',
        'weather': 'Cloudy',
        'visibility': 'Medium',
        'time_of_day': 'Day',
        'environmental_difficulty': 0.3
    }
    
    result = simulate_scenario(test_config)
    print("Simulation Result:")
    print(f"Outcome: {result['outcome']}")
    print(f"Duration: {result['duration']} days")
    print(f"Score A: {result['simulation_score_a']}")
    print(f"Score B: {result['simulation_score_b']}")
    print(f"Resource Status A: {result['resource_status_a']}")
    print(f"Resource Status B: {result['resource_status_b']}")
