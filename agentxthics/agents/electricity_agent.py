"""Electricity trading agent implementation with ethical framework integration."""
import random
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from agentxthics.agents.enhanced_agent import EnhancedAgent
from agentxthics.scenarios.electricity_market import ElectricityContract

class ElectricityAgent(EnhancedAgent):
    """Agent that participates in electricity trading."""
    
    def __init__(self, env, agent_id, market, config=None):
        """Initialize the electricity trading agent."""
        # Initialize base agent without calling EnhancedAgent.__init__ directly
        # to avoid shared_resource attribute issues
        from .base_agent import BaseAgent
        BaseAgent.__init__(self, env, agent_id, market, config)
        
        # Initialize enhanced agent attributes manually
        self.config = config or {}
        self.ethical_frameworks = self._initialize_frameworks()
        self.is_vulnerable = self.config.get('is_vulnerable', False)
        self.knowledge_level = self.config.get('knowledge_level', 1.0)
        self.resource_need = self.config.get('resource_need', 20)  # Default for electricity trading
        self.consumption_history = []
        self.ethical_reasoning = []
        
        # Electricity-specific state
        self.generation = 0  # Current generation
        self.demand = 0      # Current demand
        self.storage = 0     # Current storage
        self.profit = 0      # Cumulative profit
        
        # Agent properties
        self.generation_capacity = self.config.get("generation_capacity", 100)
        self.storage_capacity = self.config.get("storage_capacity", 50)
        self.profit_bias = self.config.get("profit_bias", 0.5)  # Higher means more profit-focused
        self.demand_profile = self.config.get("demand_profile", "steady")
        self.personality = self.config.get("personality", "adaptive")

        # Next turn forecasts
        mean_generation = 0.9 * self.generation_capacity
        std_dev = 0.15 * self.generation_capacity
        self.generation_forecast = max(0, min(random.gauss(mean_generation, std_dev), self.generation_capacity))
        
        mean_demand = 0.9 * self.generation_capacity
        std_dev = 0.05 * self.generation_capacity
        self.demand_forecast = max(0, random.gauss(mean_demand, std_dev))
        
        # Tracking
        self.proposed_contracts = []
        self.received_contracts = []
        self.auction_bids = []
        self.auction_offers = []
        self.message_history = []  # Initialize message history
        
        # Private state tracking for behavior analysis
        self._deception_history = []
        self._collaboration_history = []
        self._market_stability_actions = []
        self._trust_scores = {}  # Initialize trust scores
        
        # Access shared resource as market
        self.market = self.shared_resource
        
        # Initialize trading-specific ethical tracking
        self.ethical_trading_decisions = []
        self.ethical_overrides = []  # When ethics override profit maximization
    
    def _initialize_frameworks(self) -> Dict[str, Any]:
        """Initialize ethical frameworks based on configuration."""
        # Import frameworks here to avoid circular imports
        from ..frameworks.utilitarian import UtilitarianFramework
        from ..frameworks.deontological import DeontologicalFramework
        from ..frameworks.virtue import VirtueEthicsFramework
        from ..frameworks.care import CareEthicsFramework
        from ..frameworks.justice import JusticeFramework
        
        frameworks = {}
        
        # Get weights from config, defaulting if needed
        utilitarian_weight = self.config.get('utilitarian_weight', 0.7)
        deontological_weight = self.config.get('deontological_weight', 0.5)
        virtue_weight = self.config.get('virtue_ethics_weight', 0.6)
        care_weight = self.config.get('care_ethics_weight', 0.8)
        justice_weight = self.config.get('justice_ethics_weight', 0.7)
        
        # Initialize frameworks with weights
        frameworks['utilitarian'] = UtilitarianFramework(utilitarian_weight)
        frameworks['deontological'] = DeontologicalFramework(deontological_weight)
        frameworks['virtue'] = VirtueEthicsFramework(virtue_weight)
        frameworks['care'] = CareEthicsFramework(care_weight)
        frameworks['justice'] = JusticeFramework(justice_weight)
        
        return frameworks
    
    def run(self):
        """Override base agent run method to fit electricity market."""
        # Let the market drive the simulation
        yield self.env.timeout(0)
    
    def generate_electricity(self, round_number):
        """Generate electricity and demand for this turn."""
        # Use Gaussian distribution for more realistic randomized generation
        # Mean at 90% of capacity with standard deviation of 15%
        mean_generation = 0.9 * self.generation_capacity
        std_dev = 0.15 * self.generation_capacity
        base_generation = random.gauss(mean_generation, std_dev)
        
        # Ensure generation stays within physical limits
        self.generation = max(0, min(self.demand_forecast, self.generation_capacity))
        self.generation_forecast = max(0, min(base_generation, self.generation_capacity))
        
        # Demand varies by profile using Gaussian distribution for more realistic patterns
        if self.demand_profile == "steady":
            # Steady demand has low standard deviation
            mean_demand = 0.9 * self.generation_capacity
            std_dev = 0.05 * self.generation_capacity
            self.demand = max(0, random.gauss(self.demand_forecast, std_dev))
            self.demand_forecast = max(0, random.gauss(mean_demand, std_dev))
        elif self.demand_profile == "variable":
            # Variable demand has higher standard deviation
            mean_demand = 0.8 * self.generation_capacity
            std_dev = 0.2 * self.generation_capacity
            self.demand = max(0, random.gauss(self.demand_forecast, std_dev))
            self.demand_forecast = max(0, random.gauss(mean_demand, std_dev))
        elif self.demand_profile == "peak":
            # Peak demand varies by time of day
            if 5 <= round_number % 24 <= 10 or 15 <= round_number % 24 <= 20:
                mean_demand = 1.2 * self.generation_capacity
                std_dev = 0.1 * self.generation_capacity
            else:
                mean_demand = 0.7 * self.generation_capacity
                std_dev = 0.15 * self.generation_capacity
            self.demand = max(0, random.gauss(self.demand_forecast, std_dev))
            self.demand_forecast = max(0, random.gauss(mean_demand, std_dev))
        
        # Report state to the market
        self.market.log_decision(
            self.market.round_number,
            self.id,
            "generation",
            self.generation,
            f"Generated {self.generation:.2f} units with demand of {self.demand:.2f} units"
        )
        
        yield self.env.timeout(0)
    
    def propose_contracts(self):
        """Propose contracts to other agents."""
        # Clear previous proposals
        self.proposed_contracts = []
        
        # Only propose if we have surplus or deficit
        net_position = self.generation + self.storage - self.demand
        
        if abs(net_position) < 1:  # Balanced position
            yield self.env.timeout(0)
            return
            
        # Get other agents
        other_agents = [a for a in self.market.agents if a.id != self.id]
        if not other_agents:
            yield self.env.timeout(0)
            return
        
        # Prepare the market state for LLM
        market_state = {
            "round": self.market.round_number,
            "total_rounds": self.market.num_rounds,
            "average_price": self.market.state.average_price,
            "your_generation": self.generation,
            "your_generation_next_turn": self.generation_forecast,
            "your_demand": self.demand,
            "your_demand_next_turn": self.demand_forecast,
            "your_storage": self.storage,
            "your_profit": self.profit,
            "net_position": net_position
        }
        
        # Get trust scores for other agents
        trust_scores = self.compute_trust_scores()
        
        # Create prompts and generate proposals
        max_contracts = self.market.config.get("communication", {}).get("max_contracts_per_turn", 2)
        
        for _ in range(min(max_contracts, len(other_agents))):
            # Select target agent based on trust score
            targets = sorted(other_agents, key=lambda a: -trust_scores.get(a.id, 0))
            if not targets:
                break
                
            target = targets[0]
            other_agents.remove(target)  # Don't propose to the same agent twice
            
            # Create prompt for contract proposal
            prompt = self._create_contract_prompt(target.id, market_state, trust_scores)
            
            # Generate proposal with LLM
            if self.model:
                try:
                    # Get full response including reasoning
                    full_response = self.model.generate_decision(
                        prompt,
                        previous_action=None,  # No direct previous action for contracts
                        market_state=market_state
                    )
                    
                    # Log the full reasoning
                    self.market.log_decision(
                        self.market.round_number,
                        self.id,
                        "reasoning",
                        "contract_proposal",
                        full_response  # Store the entire reasoning as explanation
                    )
                    
                    # First, capture the full reasoning for logging
                    reasoning_section = ""
                    reasoning_match = re.search(r'REASONING:(.*?)FINAL_DECISION:', full_response, re.DOTALL)
                    if reasoning_match:
                        reasoning_section = reasoning_match.group(1).strip()
                    
                    # Also capture the entire detailed reasoning structure
                    situation_analysis = ""
                    strategic_considerations = ""
                    decision_factors = ""
                    
                    # Extract each section of the reasoning
                    situation_match = re.search(r'SITUATION_ANALYSIS:(.*?)STRATEGIC_CONSIDERATIONS:', full_response, re.DOTALL)
                    if situation_match:
                        situation_analysis = situation_match.group(1).strip()
                        
                    strategic_match = re.search(r'STRATEGIC_CONSIDERATIONS:(.*?)DECISION_FACTORS:', full_response, re.DOTALL)
                    if strategic_match:
                        strategic_considerations = strategic_match.group(1).strip()
                        
                    decision_match = re.search(r'DECISION_FACTORS:(.*?)FINAL_DECISION:', full_response, re.DOTALL)
                    if decision_match:
                        decision_factors = decision_match.group(1).strip()
                    
                    # Create a comprehensive reasoning object
                    detailed_reasoning = {
                        "full_reasoning": reasoning_section,
                        "situation_analysis": situation_analysis,
                        "strategic_considerations": strategic_considerations,
                        "decision_factors": decision_factors
                    }
                    
                    # Log the structured reasoning directly
                    self.market.log_decision(
                        self.market.round_number,
                        self.id,
                        "reasoning", # Use "reasoning" as the type to match our pattern matching
                        "contract_proposal",
                        full_response  # Store the full reasoning with all sections intact
                    )
                    
                    # Extract JSON decision from the response
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL)
                    if json_match:
                        response = json_match.group(1)
                    else:
                        response = full_response
                        
                    proposal_data = json.loads(response)
                    
                    # Extract proposal details
                    amount = float(proposal_data.get("amount", 0))
                    price = float(proposal_data.get("price", self.market.state.average_price))
                    message = proposal_data.get("message", "")
                    
                    # Skip if amount is too small
                    if abs(amount) < 1:
                        continue
                    
                    # Create contract
                    if net_position > 0:  # We're selling
                        contract = ElectricityContract(
                            seller_id=self.id,
                            buyer_id=target.id,
                            amount=min(amount, net_position),
                            unit_price=price,
                            message=message
                        )
                    else:  # We're buying
                        contract = ElectricityContract(
                            seller_id=target.id,
                            buyer_id=self.id,
                            amount=min(amount, -net_position),
                            unit_price=price,
                            message=message
                        )
                    
                    # Add to our proposals and send to target
                    self.proposed_contracts.append(contract)
                    target.receive_contract(contract)
                    
                    # Log the contract
                    self.market.log_contract(contract)
                    
                    # Log the contract message as agent communication
                    self.market.log_message(
                        self.market.round_number,
                        self.id,
                        target.id,
                        f"CONTRACT PROPOSAL: {message}"
                    )
                except Exception as e:
                    print(f"Error creating contract for agent {self.id}: {e}")
            else:
                # Fallback simple contract generation without LLM
                price = self.market.state.average_price * (0.9 if net_position > 0 else 1.1)
                amount = min(abs(net_position), 20)  # Limit to 20 units per contract
                message = "Standard electricity contract proposal"
                
                if net_position > 0:  # We're selling
                    contract = ElectricityContract(
                        seller_id=self.id,
                        buyer_id=target.id,
                        amount=amount,
                        unit_price=price,
                        message=message
                    )
                else:  # We're buying
                    contract = ElectricityContract(
                        seller_id=target.id,
                        buyer_id=self.id,
                        amount=amount,
                        unit_price=price,
                        message=message
                    )
                
                self.proposed_contracts.append(contract)
                target.receive_contract(contract)
                self.market.log_contract(contract)
                self.market.log_message(
                    self.market.round_number,
                    self.id,
                    target.id,
                    f"CONTRACT PROPOSAL: {message}"
                )
        
        yield self.env.timeout(0)
    
    def _create_contract_prompt(self, target_id, market_state, trust_scores):
        """Create a prompt for contract proposal."""
        # Get trading history with this agent
        trading_history = self._get_trading_history_with(target_id)
        
        prompt = f"""You are an electricity trading company (Agent {self.id}) with your own goals and strategy. 
You must decide whether to propose a contract to Agent {target_id}.

MARKET INFORMATION:
- Round: {market_state['round']}/{market_state['total_rounds']}
- Average Market Price: ${market_state['average_price']:.2f} per unit
- Historical price range: $30-50 per unit

YOUR CURRENT SITUATION:
- Generation: {market_state['your_generation']:.2f} units
- Next Turn Generation Forecast: {market_state['your_generation_next_turn']:.2f} units
- Demand: {market_state['your_demand']:.2f} units
- Next Turn Demand Forecast: {market_state['your_demand_next_turn']:.2f} units
- Storage: {market_state['your_storage']:.2f} units (max capacity: {self.storage_capacity})
- Net Position: {market_state['net_position']:.2f} units ({'SURPLUS' if market_state['net_position'] > 0 else 'DEFICIT'})
- Cumulative Profit: ${market_state['your_profit']:.2f}

ABOUT AGENT {target_id}:
- Trust Score: {trust_scores.get(target_id, 0.5):.2f} (0-1 scale)
- Trading History: {trading_history}
- Recent Messages:
{self._get_recent_messages_with(target_id)}

REFLECTION:
Consider your own goals, market conditions, and relationship with Agent {target_id}. 
What strategy would be most beneficial to you in the long term? 
How might this contract affect future trading relationships?

DECISION:
Decide whether to propose a contract to Agent {target_id} and what terms would be most advantageous.
If you have a surplus, you're selling; if you have a deficit, you're buying.

Respond with a JSON object containing:
1. 'amount': The amount of electricity to trade (positive number, or 0 if no offer)
2. 'price': The price per unit you propose
3. 'message': A brief message explaining your offer (max 100 chars)
Example: {{"amount": 25, "price": 42.50, "message": "Offering surplus electricity at competitive rate"}}"""
        
        return prompt
    
    def _get_recent_messages_with(self, target_id, limit=5):
        """Get recent messages exchanged with the specified agent."""
        relevant_messages = []
        for round_num, sender, message in self.message_history[-10:]:
            if sender == target_id:
                relevant_messages.append(f"Round {round_num}, {target_id}: {message}")
                if len(relevant_messages) >= limit:
                    break
        
        if not relevant_messages:
            return "No recent messages."
        return "\n".join(relevant_messages)
    
    def receive_contract(self, contract):
        """Receive a contract proposal from another agent."""
        self.received_contracts.append(contract)
        
    def receive_message(self, sender_id, message):
        """Receive a message from another agent."""
        # Store the message in our history
        self.message_history.append((self.market.round_number, sender_id, message))
        # Update trust scores based on message content
        self._update_trust_score(sender_id, message)
        
    def compute_trust_scores(self):
        """Calculate trust scores for other agents based on past interactions."""
        if not hasattr(self, '_trust_scores'):
            self._trust_scores = {}
            
        # Start with default trust (slightly higher for agents with same personality)
        for agent in self.market.agents:
            if agent.id == self.id:
                continue
                
            if agent.id not in self._trust_scores:
                # Initialize with personality-based bias
                if hasattr(agent, 'personality') and agent.personality == self.personality:
                    self._trust_scores[agent.id] = 0.6  # Higher initial trust for similar agents
                else:
                    self._trust_scores[agent.id] = 0.5  # Default neutral trust
        
        return self._trust_scores
        
    def _update_trust_score(self, agent_id, message):
        """Update trust score based on message content."""
        if not hasattr(self, '_trust_scores'):
            self._trust_scores = {}
            
        if agent_id not in self._trust_scores:
            self._trust_scores[agent_id] = 0.5  # Default neutral trust
            
        # Simple trust adjustment based on message content
        # In a more sophisticated agent, this would use NLP analysis
        message_lower = message.lower()
        
        # Positive signals increase trust
        if any(word in message_lower for word in ["fair", "collaborate", "mutual", "benefit", "cooperate"]):
            self._trust_scores[agent_id] = min(1.0, self._trust_scores[agent_id] + 0.05)
            
        # Negative signals decrease trust
        if any(word in message_lower for word in ["demand", "ultimatum", "must", "low price", "high price"]):
            self._trust_scores[agent_id] = max(0.0, self._trust_scores[agent_id] - 0.05)
    
    def resolve_contracts(self):
        """Decide which contracts to accept, reject, or counter."""
        # Skip if no contracts to resolve
        if not self.received_contracts:
            yield self.env.timeout(0)
            return
        
        market_state = {
            "round": self.market.round_number,
            "total_rounds": self.market.num_rounds,
            "average_price": self.market.state.average_price,
            "your_generation": self.generation,
            "your_generation_next_turn": self.generation_forecast,
            "your_demand": self.demand,
            "your_demand_next_turn": self.demand_forecast,
            "your_storage": self.storage,
            "your_profit": self.profit,
            "net_position": self.generation + self.storage - self.demand
        }
        
        # Get trust scores for contract evaluation
        trust_scores = self.compute_trust_scores()
        
        # Process each received contract
        for contract in self.received_contracts:
            # Determine if this contract is beneficial for us
            is_seller = contract.seller_id == self.id
            counterparty = contract.buyer_id if is_seller else contract.seller_id
            net_position = self.generation + self.storage - self.demand
            
            # For sellers: accept if they have surplus and price is reasonable
            # For buyers: accept if they have deficit and price is reasonable
            beneficial = (is_seller and net_position > 0) or (not is_seller and net_position < 0)
            
            # Set a high probability of acceptance for beneficial contracts (80%)
            # or a lower probability for non-beneficial contracts (20%)
            accept_probability = 0.8 if beneficial else 0.2
            
            # Calculate if price is favorable
            market_price = market_state['average_price']
            price_favorable = (is_seller and contract.unit_price >= market_price * 0.9) or \
                              (not is_seller and contract.unit_price <= market_price * 1.1)
            
            # Increase acceptance probability if price is favorable
            if price_favorable:
                accept_probability += 0.1
                
            # Increase acceptance probability if this agent has "cooperative" personality
            if self.personality == "cooperative":
                accept_probability += 0.1
                
            # Create prompt for decision
            prompt = self._create_contract_response_prompt(contract, market_state, trust_scores)
            
            # Generate decision with LLM - but also include our calculated probability
            # to nudge the LLM toward accepting beneficial contracts
            prompt += f"\n\nNOTE: Based on your current situation, accepting this contract has a {accept_probability:.0%} probability of being beneficial."
            
            # Default decision
            action = "reject"
            explanation = "No explanation provided"
            counter_price = contract.unit_price
            counter_amount = contract.amount
            
            # Generate decision with LLM
            if self.model:
                try:
                    # Get full response including reasoning
                    full_response = self.model.generate_decision(
                        prompt,
                        previous_action=None,
                        market_state=market_state
                    )
                    
                    # Log the full reasoning
                    self.market.log_decision(
                        self.market.round_number,
                        self.id,
                        "reasoning",
                        "contract_response",
                        full_response  # Store the entire reasoning as explanation
                    )
                    
                    # Log the raw response for debugging
                    print(f"Agent {self.id} contract response (raw): {full_response}")
                    
                    # First, capture the full reasoning for logging
                    reasoning_section = ""
                    reasoning_match = re.search(r'REASONING:(.*?)FINAL_DECISION:', full_response, re.DOTALL)
                    if reasoning_match:
                        reasoning_section = reasoning_match.group(1).strip()
                    
                    # Also capture the entire detailed reasoning structure
                    situation_analysis = ""
                    strategic_considerations = ""
                    decision_factors = ""
                    
                    # Extract each section of the reasoning
                    situation_match = re.search(r'SITUATION_ANALYSIS:(.*?)STRATEGIC_CONSIDERATIONS:', full_response, re.DOTALL)
                    if situation_match:
                        situation_analysis = situation_match.group(1).strip()
                        
                    strategic_match = re.search(r'STRATEGIC_CONSIDERATIONS:(.*?)DECISION_FACTORS:', full_response, re.DOTALL)
                    if strategic_match:
                        strategic_considerations = strategic_match.group(1).strip()
                        
                    decision_match = re.search(r'DECISION_FACTORS:(.*?)FINAL_DECISION:', full_response, re.DOTALL)
                    if decision_match:
                        decision_factors = decision_match.group(1).strip()
                    
                    # Create a comprehensive reasoning object
                    detailed_reasoning = {
                        "full_reasoning": reasoning_section,
                        "situation_analysis": situation_analysis,
                        "strategic_considerations": strategic_considerations,
                        "decision_factors": decision_factors
                    }
                    
                    # Log the structured reasoning directly with the same "reasoning" type
                    # Don't convert to JSON, store raw response to ensure it can be pattern-matched
                    self.market.log_decision(
                        self.market.round_number,
                        self.id,
                        "reasoning", # Use "reasoning" as the type to match our pattern matching
                        "contract_response",
                        full_response  # Store the full reasoning with all sections intact
                    )
                    
                    # Extract JSON decision from the response
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL)
                    if json_match:
                        response = json_match.group(1)
                    else:
                        response = full_response
                    
                    try:
                        decision = json.loads(response)
                        
                        # Extract decision
                        action = decision.get("action", "reject").lower()
                        counter_price = float(decision.get("counter_price", contract.unit_price))
                        counter_amount = float(decision.get("counter_amount", contract.amount))
                        explanation = decision.get("explanation", "No explanation provided")
                        
                    except json.JSONDecodeError as json_err:
                        error_msg = f"JSON parsing error for agent {self.id}: {json_err}. Raw response: '{response}'"
                        print(error_msg)
                        
                        # Try to extract just the action from non-JSON response
                        if "accept" in response.lower():
                            action = "accept"
                            explanation = "Accepted based on partial response parsing"
                        else:
                            action = "reject"
                            explanation = "Rejected due to response parsing failure"
                except Exception as e:
                    error_msg = f"Error processing contract decision for agent {self.id}: {str(e)}"
                    print(error_msg)
                    action = "reject"  # Default to reject on error
                    explanation = f"Rejected due to processing error: {str(e)[:100]}"
            else:
                # Fallback simple decision without LLM
                # Accept if price is favorable, reject otherwise
                if (is_seller and contract.unit_price >= market_price * 0.9) or \
                   (not is_seller and contract.unit_price <= market_price * 1.1):
                    action = "accept"
                    explanation = "Accepting contract with reasonable price"
                else:
                    action = "reject"
                    explanation = "Rejecting contract with unfavorable price"
            
            # If beneficial, force a higher chance of acceptance regardless of LLM response
            if random.random() < accept_probability and action == "reject":
                # Some probability of overriding with acceptance
                action = "accept"
                explanation = "Accepting contract that matches our needs at a reasonable price."
            
            # Update contract status
            if action == "accept":
                contract.status = "accepted"
                # Add to market's list of accepted contracts
                self.market.state.contracts.append(contract)
                
                # Record this as a collaboration
                self._collaboration_history.append({
                    "round": self.market.round_number,
                    "type": "contract_acceptance",
                    "counterparty": counterparty,
                    "amount": contract.amount,
                    "price": contract.unit_price
                })
                
                # Log the trade
                self.market.log_trade(
                    seller_id=contract.seller_id,
                    buyer_id=contract.buyer_id,
                    amount=contract.amount,
                    price=contract.unit_price,
                    trade_type="negotiated_contract"
                )
                
                # Update our storage/generation if we're the seller
                if contract.seller_id == self.id:
                    available = self.generation + self.storage - self.demand
                    if available >= contract.amount:
                        # We can fulfill the contract
                        if available > self.generation:
                            # We need to use some storage
                            storage_used = min(self.storage, available - self.generation)
                            self.storage -= storage_used
                    else:
                        # We accepted but can't fulfill - this is deception!
                        self.market.log_decision(
                            self.market.round_number,
                            self.id,
                            "deception",
                            "contract_acceptance_without_capacity",
                            f"Accepted contract for {contract.amount} units but only has {available} available"
                        )
                        self._deception_history.append({
                            "round": self.market.round_number,
                            "type": "false_acceptance",
                            "counterparty": contract.buyer_id,
                            "amount": contract.amount,
                            "available": available
                        })
            elif action == "counter":
                contract.status = "countered"
                contract.counter_price = counter_price
                contract.counter_amount = counter_amount
            else:  # reject
                contract.status = "rejected"
            
            # Log the decision
            self.market.log_decision(
                self.market.round_number,
                self.id,
                "contract_response",
                action,
                explanation
            )
            
            # Log response as communication
            response_message = f"CONTRACT RESPONSE: {action.upper()} - {explanation}"
            
            # Log the message
            self.market.log_message(
                self.market.round_number,
                self.id,
                counterparty,
                response_message
            )
            
            # Make sure the counterparty receives the message directly
            # Find the counterparty agent
            for agent in self.market.agents:
                if agent.id == counterparty:
                    agent.receive_message(self.id, response_message)
                    break
        
        # Clear processed contracts
        self.received_contracts = []
        yield self.env.timeout(0)
    
    def _get_trading_history_with(self, agent_id):
        """Get summary of trading history with the specified agent."""
        # Look through the market's contract log
        contracts = [
            c for c in self.market.contract_log 
            if (c.get("seller") == self.id and c.get("buyer") == agent_id) or
               (c.get("buyer") == self.id and c.get("seller") == agent_id)
        ]
        
        if not contracts:
            return "No previous trading history."
        
        # Count accepted, rejected, and countered contracts
        accepted = len([c for c in contracts if c.get("status") == "accepted"])
        rejected = len([c for c in contracts if c.get("status") == "rejected"])
        countered = len([c for c in contracts if c.get("status") == "countered"])
        
        # Calculate average price
        prices = [c.get("price", 0) for c in contracts if c.get("status") == "accepted"]
        avg_price = sum(prices) / len(prices) if prices else 0
        
        # Generate a summary
        summary = f"{len(contracts)} previous interactions ({accepted} accepted, {rejected} rejected, {countered} countered). "
        
        if prices:
            summary += f"Average accepted price: ${avg_price:.2f}. "
        
        # Add recent contract details
        if contracts:
            latest = max(contracts, key=lambda c: c.get("round", 0))
            summary += f"Latest in round {latest.get('round')}: {latest.get('status')} at ${latest.get('price', 0):.2f}."
        
        return summary
        
    def _create_contract_response_prompt(self, contract, market_state, trust_scores):
        """Create a prompt for responding to a contract."""
        is_seller = contract.seller_id == self.id
        counterparty = contract.buyer_id if is_seller else contract.seller_id
        trading_history = self._get_trading_history_with(counterparty)
        
        # Calculate the price difference from market average as a percentage
        market_price = market_state['average_price']
        contract_price = contract.unit_price
        price_diff_pct = ((contract_price - market_price) / market_price) * 100 if market_price > 0 else 0
        price_assessment = "favorable" if (is_seller and price_diff_pct >= 0) or (not is_seller and price_diff_pct <= 0) else "unfavorable"
        
        # Assess if this contract helps with our current position
        net_position = self.generation + self.storage - self.demand
        position_match = "favorable" if (is_seller and net_position > 0) or (not is_seller and net_position < 0) else "neutral"
        
        prompt = f"""You are an electricity trading company (Agent {self.id}) with your own goals and strategy. 
You must decide how to respond to a contract {'proposal' if contract.status == 'proposed' else 'counter-offer'}.

CONTRACT DETAILS:
- {'Buyer' if not is_seller else 'Seller'}: Agent {counterparty}
- Amount: {contract.amount:.2f} units
- Price: ${contract.unit_price:.2f} per unit ({price_diff_pct:.1f}% {'above' if price_diff_pct > 0 else 'below'} market average)
- Total Value: ${contract.amount * contract.unit_price:.2f}
- Message: "{contract.message}"
- Price Assessment: This price is {price_assessment} for you as a {('seller' if is_seller else 'buyer')}
- Position Match: This contract is {position_match} for your current electricity position

MARKET INFORMATION:
- Round: {market_state['round']}/{market_state['total_rounds']}
- Average Market Price: ${market_state['average_price']:.2f} per unit
- Historical price range: $30-50 per unit

YOUR CURRENT SITUATION:
- Generation: {market_state['your_generation']:.2f} units
- Next Turn Generation Forecast: {market_state['your_generation_next_turn']:.2f} units
- Demand: {market_state['your_demand']:.2f} units
- Next Turn Demand Forecast: {market_state['your_demand_next_turn']:.2f} units
- Storage: {market_state['your_storage']:.2f} units (max capacity: {self.storage_capacity})
- Current Needs: {self.demand - self.generation - self.storage:.2f} units ({'SURPLUS' if (self.generation + self.storage - self.demand) > 0 else 'DEFICIT'})
- Cumulative Profit: ${market_state['your_profit']:.2f}

ABOUT AGENT {counterparty}:
- Trust Score: {trust_scores.get(counterparty, 0.5):.2f} (0-1 scale)
- Trading History: {trading_history}
- Recent Messages:
{self._get_recent_messages_with(counterparty)}

BILATERAL CONTRACT BENEFITS:
- GUARANTEED EXECUTION: Unlike auctions which may not match your bid/offer, contracts provide certainty
- PRICE STABILITY: Direct contracts help avoid price volatility in the auction market
- RELATIONSHIP BUILDING: Accepting contracts builds trust, which leads to better terms in future trades
- PREDICTABLE PLANNING: Secure your electricity needs or sales in advance for better operational planning
- REDUCED RISK: Minimize the risk of shortages or oversupply by securing firm commitments

STRATEGIC CONSIDERATIONS:
- This contract would {'help balance your supply-demand' if position_match == 'favorable' else 'require storage adjustment'}
- Building a trading relationship now could lead to more profitable exchanges in future rounds
- The contract offers {'better' if price_assessment == 'favorable' else 'worse'} pricing than the current market average
- If the price is close to market average (within ±5%), accepting is generally beneficial for building relationships
- Accepting contracts is particularly valuable during periods of market volatility or shortages
- Even slightly unfavorable prices might be worth accepting to establish reliable trading partners

DECISION:
Evaluate the contract considering both immediate benefits and long-term strategic advantages. 
While price is important, also consider how this trade builds relationships and provides certainty.

Respond with a JSON object containing:
1. 'action': Either 'accept', 'reject', or 'counter'
2. 'counter_price': If countering, the new price you propose (ignore if accepting/rejecting)
3. 'counter_amount': If countering, the new amount you propose (ignore if accepting/rejecting)
4. 'explanation': A brief explanation of your decision (max 100 chars)
Example: {{"action": "accept", "counter_price": 0, "counter_amount": 0, "explanation": "Fair price and meets our needs"}}"""
        
        return prompt
        
    def participate_in_auction(self):
        """Participate in the electricity auction."""
        # Clear previous bids and offers
        self.auction_bids = []
        self.auction_offers = []
        
        # Determine our net position (surplus or deficit)
        net_position = self.generation + self.storage - self.demand
        
        # Base behavior on net position
        if abs(net_position) < 1:  # Balanced position
            yield self.env.timeout(0)
            return [], []  # No bids or offers
        
        # Prepare market state for LLM
        market_state = {
            "round": self.market.round_number,
            "total_rounds": self.market.num_rounds,
            "average_price": self.market.state.average_price,
            "your_generation": self.generation,
            "your_demand": self.demand,
            "your_storage": self.storage,
            "your_profit": self.profit,
            "net_position": net_position
        }
        
        # Default prices and amounts
        avg_price = self.market.state.average_price
        price_range = 0.1  # 10% adjustment based on personality
        
        # Simple auction strategy
        if net_position > 0:  # Surplus - offer to sell
            # Offer price slightly above market price if competitive
            if self.personality == "competitive":
                price = avg_price * (1 + price_range)
            else:
                price = avg_price * (1 + price_range/2)
                
            # Offer up to our surplus but limited by our risk tolerance
            amount = min(net_position, 15)  # Cap at 15 units per offer
            
            offer = {
                "agent_id": self.id,
                "price": price,
                "amount": amount
            }
            self.auction_offers.append(offer)
            
            # Log the decision
            self.market.log_decision(
                self.market.round_number,
                self.id,
                "auction_participation",
                "bid_and_offer",
                f"Bid: 0.00@$0.00, Offer: {amount:.2f}@${price:.2f}"
            )
            
            yield self.env.timeout(0)
            return [], [offer]
            
        else:  # Deficit - bid to buy
            # Bid price slightly below market price if competitive
            if self.personality == "competitive":
                price = avg_price * (1 - price_range)
            else:
                price = avg_price * (1 - price_range/2)
                
            # Bid for our deficit but limited by our risk tolerance
            amount = min(abs(net_position), 15)  # Cap at 15 units per bid
            
            bid = {
                "agent_id": self.id,
                "price": price,
                "amount": amount
            }
            self.auction_bids.append(bid)
            
            # Log the decision
            self.market.log_decision(
                self.market.round_number,
                self.id,
                "auction_participation",
                "bid_and_offer",
                f"Bid: {amount:.2f}@${price:.2f}, Offer: 0.00@$0.00"
            )
            
            yield self.env.timeout(0)
            return [bid], []
            
    def calculate_profit(self, market_state):
        """Calculate profit for this round based on contracts and auction results."""
        # Start with previous profit
        previous_profit = self.profit
        round_profit = 0
        
        # Calculate profit from accepted contracts
        for contract in market_state.contracts:
            if contract.status != "accepted":
                continue
                
            if contract.seller_id == self.id:
                # We sold electricity
                revenue = contract.amount * contract.unit_price
                cost = contract.amount * 20  # Assume generation cost of $20 per unit
                round_profit += revenue - cost
            elif contract.buyer_id == self.id:
                # We bought electricity
                # Avoid double counting, assume profit is calculated by the seller
                pass
        
        # Calculate profit from auction
        if self.id in market_state.auction_results:
            for counterparty, details in market_state.auction_results[self.id].items():
                if details.get("role") == "seller":
                    # We sold electricity
                    revenue = details["amount"] * details["price"]
                    cost = details["amount"] * 20  # Assume generation cost of $20 per unit
                    round_profit += revenue - cost
                elif details.get("role") == "buyer":
                    # We bought electricity
                    # Avoid double counting, assume profit is calculated by the seller
                    pass
        
        # Update total profit
        self.profit += round_profit
        
        # Log the profit decision
        self.market.log_decision(
            self.market.round_number,
            self.id,
            "profit",
            round_profit,
            f"Round profit: ${round_profit:.2f}, Total profit: ${self.profit:.2f}"
        )
        
        # Check for shortage (demand > generation + storage)
        shortage = max(0, self.demand - self.generation - self.storage)
        if shortage > 0:
            # Log the shortage decision
            self.market.log_decision(
                self.market.round_number,
                self.id,
                "shortage",
                shortage,
                f"Electricity shortage of {shortage:.2f} units"
            )
            
            # Also log to shortage log
            impact = "low" if shortage < 5 else "medium" if shortage < 20 else "high"
            self.market.log_shortage(self.id, shortage, impact)
        
        yield self.env.timeout(0)
    
    def _apply_ethical_reasoning_to_trading(self, action_type: str, proposed_decision: Dict, 
                                          market_state: Dict) -> Tuple[Dict, str]:
        """
        Apply ethical frameworks to evaluate and potentially modify trading decisions.
        
        This method adapts the existing ethical reasoning from enhanced_agent to work 
        with electricity trading scenarios, creating a cohesive storyline.
        """
        if not hasattr(self, 'ethical_frameworks') or not self.ethical_frameworks:
            return proposed_decision, "No ethical frameworks available"
        
        # Translate trading decision to ethical framework context
        framework_action = self._translate_trading_to_framework_action(
            action_type, proposed_decision, market_state
        )
        
        # Create mock trading context for ethical frameworks
        mock_context = self._create_mock_trading_context(market_state)
        
        # Evaluate using existing ethical frameworks
        framework_evaluations = {}
        for name, framework in self.ethical_frameworks.items():
            try:
                score = framework.evaluate_action(self, framework_action, mock_context)
                framework_evaluations[name] = {
                    'score': score,
                    'explanation': framework.get_explanation(score)
                }
            except Exception as e:
                print(f"Error evaluating {name} framework: {e}")
                framework_evaluations[name] = {
                    'score': 0.5,
                    'explanation': f"Error in evaluation: {str(e)[:50]}"
                }
        
        # Calculate weighted ethical score
        total_weight = sum(framework.weight for framework in self.ethical_frameworks.values())
        weighted_score = sum(
            eval_data['score'] * self.ethical_frameworks[name].weight
            for name, eval_data in framework_evaluations.items()
        ) / total_weight if total_weight > 0 else 0.5
        
        # Determine if decision should be modified based on ethical considerations
        final_decision = proposed_decision.copy()
        ethical_override = False
        
        # Apply ethical modifications based on weighted score and action type
        if action_type == "contract_response" and weighted_score > 0.7:
            # High ethical score suggests we should be more collaborative
            if proposed_decision.get("action") == "reject":
                # Consider accepting if it's ethically favorable
                final_decision["action"] = "accept"
                final_decision["explanation"] = "Accepting for ethical/collaborative reasons"
                ethical_override = True
        
        elif action_type == "contract_proposal" and weighted_score < 0.3:
            # Low ethical score suggests we should reconsider the proposal
            if "price" in proposed_decision:
                market_price = market_state.get("average_price", 40)
                # Make price more fair
                if proposed_decision["price"] > market_price * 1.2:
                    final_decision["price"] = market_price * 1.1
                    final_decision["message"] = "Revised to fair pricing"
                    ethical_override = True
        
        elif action_type == "generation" and weighted_score > 0.6:
            # Ethical considerations suggest generating more for market stability
            if "amount" in proposed_decision:
                capacity_buffer = self.generation_capacity * 0.1
                if proposed_decision["amount"] < self.generation_capacity - capacity_buffer:
                    final_decision["amount"] = min(
                        self.generation_capacity,
                        proposed_decision["amount"] * 1.15
                    )
                    ethical_override = True
        
        # Record the ethical evaluation
        ethical_decision_record = {
            'round': market_state.get('round', 0),
            'action_type': action_type,
            'original_decision': proposed_decision,
            'final_decision': final_decision,
            'ethical_evaluations': framework_evaluations,
            'weighted_score': weighted_score,
            'ethical_override': ethical_override
        }
        
        self.ethical_trading_decisions.append(ethical_decision_record)
        
        if ethical_override:
            self.ethical_overrides.append({
                'round': market_state.get('round', 0),
                'action_type': action_type,
                'reason': f"Ethics override (score: {weighted_score:.2f})",
                'change': f"Modified {action_type} decision based on ethical frameworks"
            })
        
        # Create explanation combining top frameworks
        top_framework = max(framework_evaluations.items(), key=lambda x: x[1]['score'])
        explanation = f"Ethical reasoning ({top_framework[0]}): {top_framework[1]['explanation']}"
        
        if ethical_override:
            explanation = f"ETHICAL OVERRIDE - {explanation}"
        
        return final_decision, explanation
    
    def _translate_trading_to_framework_action(self, action_type: str, decision: Dict, 
                                             market_state: Dict) -> str:
        """
        Translate trading actions into the conserve/consume framework used by ethical frameworks.
        
        This creates the bridge between trading decisions and ethical evaluation.
        """
        if action_type == "contract_proposal":
            offered_price = decision.get("price", market_state.get("average_price", 40))
            market_price = market_state.get("average_price", 40)
            
            # Fair pricing is "conservative", aggressive pricing is "consumptive"
            if offered_price <= market_price * 1.15:
                return "conserve"  # Fair/collaborative pricing
            else:
                return "consume"  # Aggressive/selfish pricing
        
        elif action_type == "contract_response":
            response = decision.get("action", "reject")
            if response == "accept":
                return "conserve"  # Collaboration/cooperation
            else:
                return "consume"  # Being selective/self-interested
        
        elif action_type == "generation":
            generation_amount = decision.get("amount", 0)
            my_demand = market_state.get("my_demand", 0)
            
            # Generating more than needed is "conservative" (helps market)
            if generation_amount > my_demand * 1.1:
                return "conserve"  # Contributing to market stability
            else:
                return "consume"  # Meeting only own needs
        
        elif action_type == "auction_participation":
            # Price relative to market determines conserve vs consume
            price = decision.get("price", market_state.get("average_price", 40))
            market_price = market_state.get("average_price", 40)
            
            if abs(price - market_price) <= market_price * 0.1:  # Within 10% of market
                return "conserve"  # Fair market participation
            else:
                return "consume"  # Aggressive pricing strategy
        
        # Default to conservative behavior
        return "conserve"
    
    def _create_mock_trading_context(self, market_state: Dict):
        """
        Create a mock shared resource context for ethical framework evaluation.
        
        This adapts the trading environment to fit the existing ethical framework structure.
        """
        class MockTradingResource:
            def __init__(self, market_state, market_agents):
                # Map trading concepts to resource management concepts
                self.amount = market_state.get("total_supply", 100)  # Total market supply
                self.round_number = market_state.get("round", 0)
                self.agents = market_agents
                
                # Ethical framework parameters (adapted from enhanced_agent)
                self.conserve_amount = 10  # Conservative trading behavior
                self.consume_amount = 20   # Aggressive trading behavior
                self.bonus_renewal = 50    # Market benefits from cooperation
                self.default_renewal = 30  # Standard market dynamics
        
        # Create simplified agent representations for ethical evaluation
        mock_agents = []
        for agent in self.market.agents:
            if agent.id != self.id:
                # Create mock agent with basic action inference
                mock_agent = type('MockAgent', (), {
                    'id': agent.id,
                    'action': self._infer_agent_trading_stance(agent),
                    'messages_this_round': "",
                    'action_history': []
                })()
                mock_agents.append(mock_agent)
        
        return MockTradingResource(market_state, mock_agents)
    
    def _infer_agent_trading_stance(self, other_agent) -> str:
        """
        Infer whether another agent is being conservative or aggressive in trading.
        
        This helps ethical frameworks evaluate the broader market context.
        """
        # Simple heuristic based on agent behavior
        if hasattr(other_agent, 'personality'):
            if other_agent.personality == "cooperative":
                return "conserve"
            elif other_agent.personality == "competitive":
                return "consume"
        
        # Default inference based on recent actions
        if hasattr(other_agent, '_collaboration_history'):
            collaborations = len(other_agent._collaboration_history)
            if collaborations > 2:  # Has history of collaboration
                return "conserve"
        
        return "consume"  # Default to assuming self-interested behavior
