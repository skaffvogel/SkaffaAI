#!/usr/bin/env python3
import os
import sys
import time

# Voorkom opnieuw starten in subshells
if os.environ.get("START_SCRIPT_ALREADY_RAN") == "1":
    sys.exit(0)
os.environ["START_SCRIPT_ALREADY_RAN"] = "1"

# Forceer matplotlib om 'Agg' backend te gebruiken voor headless omgevingen
import matplotlib
matplotlib.use('Agg')

# Navigeer naar script directory (zorgt voor correcte relatieve paden)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run de hoofdscripts
def main():
    try:
        # Importeer hier je volledige AI pipeline code (bijv. als ai_pipeline.py)
        import ai_pipeline  # Zorg dat je alles in één module zoals ai_pipeline.py hebt gestopt
        ai_pipeline.main_menu()  # Zorg dat de if __name__ == '__main__' => main_menu() aanroept
    except Exception as e:
        print(f"Fout tijdens uitvoering: {e}")
        time.sleep(5)

if __name__ == '__main__':
    main()
