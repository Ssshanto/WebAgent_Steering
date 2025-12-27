from trl import SFTConfig, SFTTrainer
import inspect

print("SFTConfig Args:")
print(inspect.signature(SFTConfig.__init__))

print("\nSFTTrainer Args:")
print(inspect.signature(SFTTrainer.__init__))
