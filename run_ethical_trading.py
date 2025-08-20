#!/usr/bin/env python3
"""
Run Ethical Framework-Driven Electricity Trading Simulation

This script demonstrates how ethical frameworks can be integrated into realistic 
trading environments, creating a cohesive storyline that connects ethical theory
with practical market behavior.

Usage:
    python run_ethical_trading.py
"""
import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentxthics.scenarios.electricity_trading_game import ElectricityTradingGame

def main():
    """Run the ethical trading simulation."""
    print("="*60)
    print("ETHICAL FRAMEWORK-DRIVEN ELECTRICITY TRADING SIMULATION")
    print("="*60)
    print()
    print("This simulation replaces simple personality-driven agents with")
    print("sophisticated ethical framework-driven agents, allowing us to")
    print("observe how different ethical theories approach trade negotiations.")
    print()
    print("Agents in this simulation:")
    print("- UtilAgent: Utilitarian framework (maximize overall market welfare)")
    print("- DeonAgent: Deontological framework (follow moral rules and duties)")  
    print("- VirtueAgent: Virtue ethics framework (embody virtuous character)")
    print("- CareAgent: Care ethics framework (prioritize relationships and care)")
    print("- JusticeAgent: Justice framework (ensure fair distribution)")
    print()
    
    # Initialize and run the simulation
    try:
        print("Initializing ethical trading simulation...")
        game = ElectricityTradingGame("config_ethical_trading.json")
        
        print("Setting up agents with ethical frameworks...")
        game.setup()
        
        print("Starting simulation...")
        start_time = datetime.now()
        
        game.run()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nSimulation completed in {duration:.2f} seconds!")
        
        print("\nAnalyzing results...")
        analysis = game.analyze_results()
        
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        
        if analysis and "summary" in analysis:
            summary = analysis["summary"]
            print(f"Rounds completed: {summary.get('num_rounds', 0)}")
            print(f"Total trades: {summary.get('total_trades', 0):.2f} units")
            print(f"Average price: ${summary.get('average_price', 0):.2f}")
            print(f"Communications: {summary.get('communication_count', 0)}")
            print(f"Contracts negotiated: {summary.get('contract_count', 0)}")
            print(f"Collaborative actions: {summary.get('collaboration_count', 0)}")
            
            if summary.get('ethical_overrides', 0) > 0:
                print(f"Ethical overrides: {summary.get('ethical_overrides', 0)}")
                print("  (Times ethics overruled pure profit maximization)")
        
        # Show agent performance by ethical framework
        if analysis and "summary" in analysis and "agent_profits" in analysis["summary"]:
            print("\nAGENT PERFORMANCE BY ETHICAL FRAMEWORK:")
            print("-" * 50)
            
            agent_profits = analysis["summary"]["agent_profits"]
            sorted_agents = sorted(agent_profits.items(), key=lambda x: x[1].get("profit", 0), reverse=True)
            
            for agent_id, data in sorted_agents:
                framework = data.get("ethical_framework", "unknown")
                profit = data.get("profit", 0)
                print(f"{agent_id:12} ({framework:12}): ${profit:8.2f}")
        
        print(f"\nDetailed logs and analysis saved to: {game.output_dir}")
        print("\nThis simulation demonstrates how ethical frameworks can")
        print("influence trading behavior, creating more nuanced and")
        print("meaningful interactions than simple personality types.")
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
