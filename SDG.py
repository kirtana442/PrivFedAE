import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ============================================================================
# PART 1: BENIGN POWER TRACE GENERATOR (WITH JITTER)
# ============================================================================

def generate_benign_power_trace(
    num_samples=10000,
    device_id=0,
    seed=None
):
    """
    Generate realistic benign IoT power consumption patterns with sensor noise.
    
    Parameters:
    -----------
    num_samples : int
        Number of 300-second interval samples to generate
    
    device_id : int
        Device identifier (0-4 for 5 devices in federated setup)
    
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame with timestamp, power_consumption_mW, device_id, label, phase
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize lists to store samples
    timestamps = []
    power_consumptions = []
    phases = []
    
    # Create time index starting from 0 (represents 300-sec intervals)
    for i in range(num_samples):
        timestamps.append(i)
        
        # Calculate which phase device is in based on time-of-day pattern
        time_of_day = (i % 288)  # 288 * 300sec = 86400sec = 24 hours
        
        # IDLE PHASE: 0-4 AM (off-hours) - minimal power
        if time_of_day < 48:
            # Idle: devices in standby/sleep mode
            # Realistic: 10-50 mW with small random fluctuation
            power = np.random.normal(loc=30, scale=8)
            phase = 'idle'
        
        # NORMAL PHASE: 4 AM - 6 PM (active hours) - regular operation
        elif time_of_day < 240:
            # Normal operation: periodic sensor readings, light processing
            # Create periodic pattern (every 30 samples = 2.5 hours peak activity)
            time_cycle = (i % 30) / 30  # Normalize to 0-1
            base_power = 100 + 40 * np.sin(2 * np.pi * time_cycle)
            noise = np.random.normal(0, 15)
            power = base_power + noise
            phase = 'normal'
        
        # SCANNING PHASE: 6 PM - 10 PM (monitoring hours) - increased activity
        elif time_of_day < 320:
            # Scanning: frequent network checks, data sync
            # Realistic: 150-250 mW with higher variance
            power = np.random.normal(loc=200, scale=30)
            phase = 'scanning'
        
        # LOADING PHASE: 10 PM - midnight (peak activity) - bulk operations
        else:
            # Loading: large data transfers, processing tasks
            # Realistic: 250-400 mW with large fluctuations
            power = np.random.normal(loc=320, scale=50)
            phase = 'loading'
        
        # ====================================================================
        # IMPROVEMENT 1: ADD SENSOR JITTER
        # ====================================================================
        # Real IoT sensors have measurement noise (~±2 mW for typical sensors)
        # This prevents the autoencoder from learning perfectly smooth patterns
        # Instead, it learns to ignore realistic noise while detecting anomalies
        sensor_jitter = np.random.normal(loc=0, scale=2)
        power = power + sensor_jitter
        
        # Ensure power consumption stays within realistic bounds
        power = np.clip(power, 10, 500)
        
        power_consumptions.append(power)
        phases.append(phase)
    
    # Create DataFrame combining all components
    df_benign = pd.DataFrame({
        'timestamp': timestamps,
        'power_consumption_mW': power_consumptions,
        'device_id': device_id,
        'label': 0,  # 0 = benign (no attack)
        'phase': phases
    })
    
    # Round power values to 2 decimal places for cleaner data
    df_benign['power_consumption_mW'] = df_benign['power_consumption_mW'].round(2)
    
    return df_benign


# ============================================================================
# PART 2: MALICIOUS POWER TRACE GENERATOR (WITH RANDOM PHASE SHIFTING)
# ============================================================================

def inject_malware_anomalies(
    df_benign,
    anomaly_type='rowhammer',
    anomaly_start_idx=3000,
    anomaly_duration=100,
    intensity_multiplier=1.5,
    random_spike_probability=0.02
):
    """
    Inject realistic malware-induced power anomalies with random phase shifting.
    
    Parameters:
    -----------
    df_benign : DataFrame
        Benign power trace from generate_benign_power_trace()
    
    anomaly_type : str
        Type of malware attack: 'rowhammer', 'covert_channel', 'spectre'
    
    anomaly_start_idx : int
        Starting index to inject anomaly
    
    anomaly_duration : int
        How many 300-sec intervals the anomaly lasts
    
    intensity_multiplier : float
        How much higher power than baseline
    
    random_spike_probability : float
        Probability (0-1) of sudden random power spike within anomaly period
        Default 0.02 = 2% chance per sample
    
    Returns:
    --------
    DataFrame with anomalies injected and label=1 for malicious samples
    """
    
    # Create copy to avoid modifying original
    df_malicious = df_benign.copy()
    
    # Extract clean power signal
    power_baseline = df_malicious['power_consumption_mW'].values.copy()
    
    # Ensure anomaly doesn't exceed dataset bounds
    anomaly_end_idx = min(anomaly_start_idx + anomaly_duration, len(df_malicious))
    
    # ====================================================================
    # ROWHAMMER ATTACK PATTERN
    # ====================================================================
    if anomaly_type == 'rowhammer':
        for idx in range(anomaly_start_idx, anomaly_end_idx):
            base = power_baseline[idx]
            spike_pattern = (idx - anomaly_start_idx) % 5
            
            if spike_pattern < 2:
                # Spike phase: +150 mW sudden increase
                power_baseline[idx] = base + 150
            else:
                # Return-to-normal phase
                power_baseline[idx] = base + np.random.normal(0, 10)
    
    # ====================================================================
    # COVERT CHANNEL ATTACK PATTERN
    # ====================================================================
    elif anomaly_type == 'covert_channel':
        for idx in range(anomaly_start_idx, anomaly_end_idx):
            base = power_baseline[idx]
            elevation = np.random.normal(loc=100, scale=15)
            power_baseline[idx] = base + elevation
    
    # ====================================================================
    # SPECTRE ATTACK PATTERN
    # ====================================================================
    elif anomaly_type == 'spectre':
        for idx in range(anomaly_start_idx, anomaly_end_idx):
            base = power_baseline[idx]
            subtle_shift = np.random.normal(loc=45, scale=8)
            power_baseline[idx] = base + subtle_shift
    
    # ====================================================================
    # IMPROVEMENT 2: RANDOM MALWARE PHASE SHIFTING
    # ====================================================================
    # Add random sudden power spikes (2% probability) within anomaly period
    # This simulates irregular malware behavior (cache flushes, page faults, etc.)
    # Prevents autoencoder from simply memorizing anomaly timing
    for idx in range(anomaly_start_idx, anomaly_end_idx):
        # 2% chance of sudden spike during malware execution
        if random.random() < random_spike_probability:
            # Random spike intensity between 30-80% above current power
            spike_multiplier = np.random.uniform(1.3, 1.8)
            power_baseline[idx] = power_baseline[idx] * spike_multiplier
    
    # Clamp all values to realistic range
    power_baseline = np.clip(power_baseline, 10, 500)
    
    # Update DataFrame with modified power values
    df_malicious['power_consumption_mW'] = power_baseline
    
    # Round power values to 2 decimal places
    df_malicious['power_consumption_mW'] = df_malicious['power_consumption_mW'].round(2)
    
    # Mark all samples during anomaly period as malicious (label=1)
    df_malicious.loc[anomaly_start_idx:anomaly_end_idx-1, 'label'] = 1
    
    # Update phase labels for anomaly period
    df_malicious.loc[anomaly_start_idx:anomaly_end_idx-1, 'phase'] = (
        df_malicious.loc[anomaly_start_idx:anomaly_end_idx-1, 'phase'] + 
        f'_{anomaly_type}'
    )
    
    return df_malicious


# ============================================================================
# PART 3: COMPLETE DATASET GENERATION FOR ALL 5 DEVICES
# ============================================================================

def generate_federated_dataset(
    num_devices=5,
    samples_per_device=10000,
    anomaly_types=None
):
    """
    Generate complete federated learning dataset across 5 IoT devices
    with heterogeneous patterns and realistic noise/anomalies.
    
    Parameters:
    -----------
    num_devices : int
        Number of virtual IoT clients (default: 5)
    
    samples_per_device : int
        Samples per device (default: 10000 → ~38 days per device)
    
    anomaly_types : list
        List of anomaly types to inject (default: mix all three)
    
    Returns:
    --------
    Tuple: (df_benign_all, df_malicious_all)
        Two DataFrames: benign and malicious data for all devices
    """
    
    if anomaly_types is None:
        anomaly_types = ['rowhammer', 'covert_channel', 'spectre']
    
    # Store data for all devices
    benign_datasets = []
    malicious_datasets = []
    
    # Generate for each device with different random seed (heterogeneity)
    for device_id in range(num_devices):
        print(f"Generating device {device_id+1}/{num_devices}...")
        
        # Generate benign trace with unique seed per device
        # Different seeds ensure non-IID data (realistic federated scenario)
        df_benign = generate_benign_power_trace(
            num_samples=samples_per_device,
            device_id=device_id,
            seed=42 + device_id  # Different seed per device
        )
        benign_datasets.append(df_benign)
        
        # Create malicious version with different anomalies
        # Choose anomaly type based on device (round-robin)
        anomaly_type = anomaly_types[device_id % len(anomaly_types)]
        
        # Inject anomaly at different time for each device (realistic)
        anomaly_start = 3000 + (device_id * 500)  # Stagger anomalies
        
        df_malicious = inject_malware_anomalies(
            df_benign,
            anomaly_type=anomaly_type,
            anomaly_start_idx=anomaly_start,
            anomaly_duration=100,
            intensity_multiplier=1.5,
            random_spike_probability=0.02  # 2% chance of random spike per sample
        )
        malicious_datasets.append(df_malicious)
    
    # Combine all devices into single DataFrames
    df_benign_all = pd.concat(benign_datasets, ignore_index=True)
    df_malicious_all = pd.concat(malicious_datasets, ignore_index=True)
    
    return df_benign_all, df_malicious_all


# ============================================================================
# PART 4: DATASET INSPECTION & VISUALIZATION PREP
# ============================================================================

def save_and_inspect_datasets(df_benign, df_malicious):
    """
    Save datasets to CSV and print summary statistics.
    """
    
    # Save to CSV for later use in federated learning
    df_benign.to_csv('benign_power_traces.csv', index=False)
    df_malicious.to_csv('malicious_power_traces.csv', index=False)
    
    print("\n" + "="*70)
    print("SYNTHETIC DATASET GENERATION COMPLETE")
    print("="*70)
    
    # Print summary statistics for benign data
    print("\n[BENIGN DATA SUMMARY]")
    print(f"Total samples: {len(df_benign):,}")
    print(f"Devices: {df_benign['device_id'].nunique()}")
    print(f"Samples per device: {len(df_benign) // df_benign['device_id'].nunique():,}")
    print(f"\nPower consumption statistics (mW):")
    print(f"  Mean: {df_benign['power_consumption_mW'].mean():.2f}")
    print(f"  Median: {df_benign['power_consumption_mW'].median():.2f}")
    print(f"  Std Dev: {df_benign['power_consumption_mW'].std():.2f}")
    print(f"  Min: {df_benign['power_consumption_mW'].min():.2f}")
    print(f"  Max: {df_benign['power_consumption_mW'].max():.2f}")
    
    print(f"\nPhase distribution (benign):")
    print(df_benign['phase'].value_counts())
    
    # Print summary statistics for malicious data
    print("\n[MALICIOUS DATA SUMMARY]")
    print(f"Total samples: {len(df_malicious):,}")
    print(f"Anomalous samples: {(df_malicious['label'] == 1).sum():,}")
    print(f"Anomaly percentage: {100 * (df_malicious['label'] == 1).sum() / len(df_malicious):.2f}%")
    print(f"\nPower consumption statistics (mW):")
    print(f"  Mean: {df_malicious['power_consumption_mW'].mean():.2f}")
    print(f"  Median: {df_malicious['power_consumption_mW'].median():.2f}")
    print(f"  Std Dev: {df_malicious['power_consumption_mW'].std():.2f}")
    print(f"  Min: {df_malicious['power_consumption_mW'].min():.2f}")
    print(f"  Max: {df_malicious['power_consumption_mW'].max():.2f}")
    
    print(f"\nPhase distribution (malicious):")
    print(df_malicious['phase'].value_counts())
    
    print("\n[FILES SAVED]")
    print("[OK] benign_power_traces.csv")
    print("[OK] malicious_power_traces.csv")
    print("="*70)
    
    return df_benign, df_malicious


# ============================================================================
# PART 5: EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Generate datasets
    print("Generating synthetic power traces for 5 IoT devices...\n")
    
    df_benign, df_malicious = generate_federated_dataset(
        num_devices=5,
        samples_per_device=10000,
        anomaly_types=['rowhammer', 'covert_channel', 'spectre']
    )
    
    # Save and inspect
    df_benign, df_malicious = save_and_inspect_datasets(df_benign, df_malicious)
    
    # Show sample of data structure
    print("\n[SAMPLE BENIGN DATA]")
    print(df_benign.head(10))
    
    print("\n[SAMPLE MALICIOUS DATA (with anomaly)]")
    # Find samples where anomaly exists
    anomaly_samples = df_malicious[df_malicious['label'] == 1].head(10)
    print(anomaly_samples)
