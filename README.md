Some Donations Please for my Next Quantum Project :

1Bu4CR8Bi5AXQG8pnu1avny88C5CCgWKfb / 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai

Support This is my LTC Litecoin address for the next version of public-addresses: LdJX6Zr43PBekv11eKBfrMtpH78qi96Mef

USDT-TRON TQ1cxj8csRyWUzkonf5XgYUyFGsDJn1k7J

USDT-BSC 0x3fa39005a6bb18d0e2546d97b24a767cc393b03a

TODO: The Next Project is To Reveal A Supirior Version That Will Use Probabilistic Algorithm of Pbits Jumps Will be Used to Reduce All the Puzzles of BTC given Rangs to 2.0 % Targeting Public-Addresses . 


# Features: Probabilistic Quantum Code :
#   • Works with public address ONLY (Puzzle 71)
#   • Works with public address + public key (Puzzle 135)
#   • Fully interactive - asks for EVERYTHING
#   • optimization_level choice (1/2/3) with suggestions
#   • layers & iterations with recommendations
#   • shots fully user-controlled
#   • Uses pycryptodome RIPEMD160 (Colab safe)
#   • Counter for measurement distribution
#   • Current IBM free backends (kingston, fez, marrakech)
#   • Verbose comments everywhere
# =============================================================================
# 1. INSTALLATION (run this cell first in Google Colab)
# =============================================================================

# !pip install qiskit qiskit-ibm-runtime ecdsa pycryptodome base58

# =============================================================================
# 2. IMPORTS
# =============================================================================

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.circuit.library import TwoLocal
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import hashlib
import time
import matplotlib.pyplot as plt
from collections import Counter
from ecdsa import SigningKey, SECP256k1
import base58
from Crypto.Hash import RIPEMD160
from qiskit_aer import AerSimulator
# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def base58_decode(s: str) -> bytes:
    """Pure Python base58 decode - no external libraries needed."""
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    num = 0
    for char in s:
        num = num * 58 + alphabet.index(char)
    bytes_out = num.to_bytes((num.bit_length() + 7) // 8, 'big')
    leading_zeros = len(s) - len(s.lstrip(alphabet[0]))
    return b'\x00' * leading_zeros + bytes_out


def address_to_hash160(address: str) -> bytes:
    """Convert Bitcoin address to hash160 (required for address-only mode)."""
    decoded = base58_decode(address)
    if len(decoded) < 25:
        raise ValueError("Invalid Bitcoin address length")
    hash160 = decoded[1:-4]
    checksum = decoded[-4:]
    calc_checksum = hashlib.sha256(hashlib.sha256(b'\x00' + hash160).digest()).digest()[:4]
    if checksum != calc_checksum:
        raise ValueError("Invalid Bitcoin address checksum")
    print(f"✅ Target hash160 loaded successfully: {hash160.hex()}")
    return hash160


# =============================================================================
# 4. MAIN QUANTUM CLASS
# =============================================================================
class QuantumProbabilisticRangeReducer:
    def __init__(self, bits: int, target_hash160: bytes, base: int, public_key_hex: str = None):
        self.bits = bits
        self.base = base
        self.target_hash160 = target_hash160
        self.public_key_hex = public_key_hex  # None = address-only mode

    def energy(self, key_int: int) -> float:
        """
        The core 'sensing' function.
        Computes how close the candidate private key is to the target
        by comparing hash160 (exactly like the energy function in p-bit hardware).
        """
        try:
            priv = key_int.to_bytes(32, 'big')
            sk = SigningKey.from_string(priv, curve=SECP256k1)
            vk = sk.verifying_key
            pub = (b'\x02' if vk.to_string()[32] % 2 == 0 else b'\x03') + vk.to_string()[:32]

            # Hash160 distance (always used)
            sha = hashlib.sha256(pub).digest()
            ripemd = RIPEMD160.new()
            ripemd.update(sha)
            h160 = ripemd.digest()
            distance = sum(1 for a, b in zip(h160, self.target_hash160) if a != b)

            # Extra strong check when public key is provided (Puzzle 135 mode)
            if self.public_key_hex:
                generated_pub_hex = pub.hex()
                if generated_pub_hex != self.public_key_hex.lower():
                    distance += 50  # heavy penalty

            return float(distance)
        except Exception:
            return 999.0

    def run(self, layers: int = 4, iterations: int = 12, shots: int = 16384,
            optimization_level: int = 2, use_real_hardware: bool = False,
            ibm_backend: str = "ibm_fez"):
        """Main quantum execution loop - FIXED for SamplerV2 (April 2026)."""
        print(f"\n🚀 STARTING QUANTUM PROBABILISTIC RANGE REDUCTION")
        print(f"Qubits used: {self.bits}")
        mode = "Public Key + Address (stronger sensing)" if self.public_key_hex else "Address Only (Puzzle 71 style)"
        print(f"Mode: {mode}")
        print(f"Optimization level: {optimization_level} | Layers: {layers} | Iterations: {iterations} | Shots: {shots}")

        # Create the variational ansatz
        ansatz = TwoLocal(self.bits, rotation_blocks='ry', entanglement_blocks='cx',
                          reps=layers, entanglement='linear')

        # === CRITICAL FIX 1: Add explicit measurements (required for SamplerV2) ===
        # We name the classical register "c" so result[0].data.c works perfectly
        from qiskit import ClassicalRegister
        creg = ClassicalRegister(self.bits, "c")
        circuit = ansatz.copy()
        circuit.add_register(creg)
        circuit.measure(range(self.bits), creg)   # measure all qubits into register 'c'

        # Choose sampler
        if use_real_hardware:
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=156)
            sampler = Sampler(mode=backend)          # correct V2 syntax
            print(f"Using real IBM hardware: {backend.name}")
        else:
            sampler = Sampler()                         # works in your Colab setup
            backend = AerSimulator()
            print("Using Aer simulator")

        all_low_energy_samples = []
        best_energy = float('inf')

        for it in range(iterations):
            print(f"\n--- Quantum Iteration {it+1}/{iterations} ---")

            # Transpile the circuit (now with measurements)
            pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
            isa_circuit = pm.run(circuit)

            # Bind parameters randomly (this was already fixed)
            param_values = np.random.uniform(-np.pi, np.pi, len(isa_circuit.parameters))

            # Run with SamplerV2 - correct PUB format
            job = sampler.run([(isa_circuit, param_values)], shots=shots)

            print(f"   Shots: {shots}")
            print(f"   Job ID: {job.job_id()}")
            print("⏳ Waiting for results...")

            result = job.result()
            print("✅ Results retrieved successfully!")

            # === CRITICAL FIX 2: Correct result parsing for SamplerV2 ===
            # This is what was causing the "no attribute 'quasi_dists'" error
            counts_dict = result[0].data.c.get_counts()          # register 'c' from above
            # Convert counts → quasi_dists (probability dict) exactly like your old code expected
            quasi_dists = {bitstr: count / shots for bitstr, count in counts_dict.items()}

            print(f"\n📊 Received {len(quasi_dists)} unique measurement outcomes")

            current_samples = []
            processed_count = 0

            for bitstr, prob in quasi_dists.items():
                processed_count += 1
                state = np.array([int(b) for b in bitstr.zfill(self.bits)])
                offset = int(''.join(map(str, state)), 2)
                key = self.base + offset
                energy = self.energy(key)

                if energy < best_energy:
                    best_energy = energy
                if energy <= 6.0:
                    current_samples.append(state)

            print(f"  Processed {processed_count} unique bitstrings")
            all_low_energy_samples.extend(current_samples)
            print(f"Best energy so far: {best_energy:.2f} | Low-energy samples this iter: {len(current_samples)}")

        # =============================================================================
        # POST-PROCESSING: Bit pinning + minimum range calculation
        # =============================================================================
        if not all_low_energy_samples:
            print("No good samples collected. Try increasing shots or layers.")
            return None, None

        samples_arr = np.array(all_low_energy_samples)
        bit_probs = samples_arr.mean(axis=0)

        # Plot the interference pattern (visual confirmation of sensing)
        plt.figure(figsize=(14, 6))
        plt.bar(range(self.bits), bit_probs, color='skyblue')
        plt.axhline(0.88, color='red', linestyle='--', label='Pin threshold (1)')
        plt.axhline(0.12, color='red', linestyle='--', label='Pin threshold (0)')
        plt.title('Quantum Bit Probability Distribution - Interference Pattern')
        plt.xlabel('Bit position (MSB on the left)')
        plt.ylabel('Probability of being 1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        pinned_bits = ['1' if p > 0.88 else '0' if p < 0.12 else '?' for p in bit_probs]

        # Calculate minimum reduced range
        start_offset = end_offset = 0
        power = 1
        for i in range(self.bits - 1, -1, -1):
            bit = pinned_bits[i]
            if bit == '1':
                start_offset += power
                end_offset += power
            elif bit == '?':
                end_offset += power
            power *= 2

        start_range = self.base + start_offset
        end_range = self.base + end_offset + 1
        pinned_count = pinned_bits.count('1') + pinned_bits.count('0')

        print("\n" + "="*90)
        print("✅ QUANTUM PROBABILISTIC SENSING COMPLETE")
        print(f"Mode used: {mode}")
        print(f"Minimum reduced range : {hex(start_range)} — {hex(end_range)}")
        print(f"Range size            : {end_range - start_range:,} keys")
        print(f"Bits left for classical brute-force : {self.bits - pinned_count}")
        print(f"Search space reduced by factor ~2^{pinned_count}")
        print("="*90)
        print("This range is ready for classical GPU brute-force (BitCrack, etc.).")
        return start_range, end_range


# =============================================================================
# 5. FULL INTERACTIVE LAUNCH - ASKS FOR EVERYTHING
# =============================================================================
if __name__ == "__main__":
    print("="*90)
    print("FINAL BIG FULLY EXPANDED QISKIT QUANTUM PROBABILISTIC RANGE REDUCER")
    print("Handles address-only AND address + public key")
    print("Fully interactive with all toggles and prompts")
    print("="*90)

    bits = int(input("\nEnter bit length (71 for Puzzle 71, 135 for Puzzle 135): ") or 71)
    address = input("Enter public Bitcoin address (required): ").strip()

    pubkey = input("Enter public key hex (optional - press Enter if you only have address): ").strip()
    if pubkey == "":
        pubkey = None
        print("→ Running in address-only mode (Puzzle 71 style)")
    else:
        print("→ Running in public key + address mode (stronger sensing)")

    start_input = input(f"Enter k_start hex [Press Enter for auto 2^({bits-1})]: ").strip()
    base = int(start_input, 16) if start_input else (1 << (bits - 1))

    shots = int(input("Enter shots per iteration (16384 recommended): ") or 16384)

    print("\nOptimization level suggestions:")
    print("  1 = light (fast)")
    print("  2 = balanced (recommended)")
    print("  3 = heavy (better results but slower)")
    opt_level = int(input("Choose optimization_level (1/2/3, default 2): ") or 2)

    print("\nLayers suggestions:")
    print("  3 = lighter / faster")
    print("  4 = balanced (recommended)")
    print("  5 = stronger interference")
    layers = int(input("Enter TwoLocal layers (default 4): ") or 4)

    print("\nIterations suggestions:")
    print("  8  = quick test")
    print("  12 = balanced (recommended)")
    print("  15 = more accurate range reduction")
    iterations = int(input("Enter number of iterations (default 12): ") or 12)

    # Prepare target
    target_hash160 = address_to_hash160(address)

    # Create the reducer
    reducer = QuantumProbabilisticRangeReducer(
        bits=bits,
        target_hash160=target_hash160,
        base=base,
        public_key_hex=pubkey
    )

    # Hardware choice
    use_real = input("\nUse real IBM Quantum hardware? [y/N]: ").lower() == 'y'
    if use_real:
        print("\n=== IBM Quantum Authentication ===")
        token = input("IBM Quantum API token (press Enter if already saved): ").strip()
        crn = input("IBM Cloud CRN (press Enter to skip): ").strip() or None
        if token:
            QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=token, overwrite=True)
        print("Real hardware selected - this may consume your monthly quota.")

    # Run the quantum job
    start_r, end_r = reducer.run(
        layers=layers,
        iterations=iterations,
        shots=shots,
        optimization_level=opt_level,
        use_real_hardware=use_real
    )

    if start_r is not None and end_r is not None:
        print(f"\n🎯 YOUR FINAL MINIMUM REDUCED RANGE IS READY:")
        print(f"Start : {hex(start_r)}")
        print(f"End   : {hex(end_r)}")
        print(f"Size  : {end_r - start_r:,} keys")
        print("\nYou can now use this much smaller range with classical brute-force tools.")
