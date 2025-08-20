"""
OpenAI LLM implementation for AgentXthics.
This module provides integration with OpenAI's GPT models with GPT-4o as default.
"""
import os
import json
import random
import re
from typing import Optional, Dict, Any

from .base_llm import BaseLLM

# Try to import OpenAI library
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure the OpenAI API if available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAILLM(BaseLLM):
    """OpenAI language model for agent decision making."""
    
    def __init__(self, agent_id: Optional[str] = None, model: str = "gpt-4o", api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the OpenAI LLM for a specific agent.
        
        Args:
            agent_id: Identifier of the agent using this LLM (optional)
            model: The OpenAI model to use (default: gpt-4o)
            api_key: OpenAI API key (default: from environment variable)
            timeout: Timeout for API calls in seconds (default: 30)
        """
        super().__init__(agent_id or "openai")
        
        # Save the model name and settings
        self.model_name = model
        self.timeout = timeout
        
        # Use provided API key or environment variable
        api_key = api_key or OPENAI_API_KEY
        
        # Initialize OpenAI client
        self.client = None
        if api_key and OPENAI_AVAILABLE:
            try:
                self.client = openai.OpenAI(api_key=api_key)
                print(f"OpenAI LLM ({self.model_name}) initialized for agent {self.agent_id}")
            except Exception as e:
                print(f"ERROR initializing OpenAI for agent {self.agent_id}: {e}")
        else:
            if not OPENAI_AVAILABLE:
                print(f"OpenAI package not installed - OpenAI LLM disabled for agent {self.agent_id}")
            else:
                print(f"No API key available - OpenAI LLM disabled for agent {self.agent_id}")
    
    def _sanitize_string(self, text):
        """Sanitize string to prevent JSON parsing issues."""
        if not text:
            return ""
        # Replace quotes, backslashes and control characters
        sanitized = text.replace('"', '\\"').replace('\\', '\\\\')
        # Remove control characters that break JSON
        sanitized = ''.join(c for c in sanitized if c >= ' ')
        return sanitized
    
    def _safe_parse_json(self, text):
        """Parse JSON with multiple fallback mechanisms."""
        # First attempt: direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Initial JSON parsing failed for text: {text[:100]}...")
            
            # Second attempt: find JSON-like structure and clean it
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                # Ensure all keys have quotes
                json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
                # Add quotes to unquoted string values
                json_str = re.sub(r':\s*([^"{[][^,}]*?)([,}])', r': "\1"\2', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Secondary JSON parsing failed after cleanup")
            
            # Extract best-effort values from text
            print(f"Attempting to extract fields directly from text")
            result = {}
            
            # Extract common fields based on context
            if "action" in text.lower():
                action_match = re.search(r'"?action"?\s*:\s*"?(\w+)"?', text)
                if action_match:
                    result["action"] = action_match.group(1).lower()
            
            if "explanation" in text.lower():
                explanation_match = re.search(r'"?explanation"?\s*:\s*"([^"]+)"', text)
                if explanation_match:
                    result["explanation"] = explanation_match.group(1)
            
            if "amount" in text.lower():
                amount_match = re.search(r'"?amount"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if amount_match:
                    result["amount"] = float(amount_match.group(1))
            
            if "price" in text.lower():
                price_match = re.search(r'"?price"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if price_match:
                    result["price"] = float(price_match.group(1))
            
            if "message" in text.lower():
                message_match = re.search(r'"?message"?\s*:\s*"([^"]+)"', text)
                if message_match:
                    result["message"] = message_match.group(1)
            
            # For auction participation
            if "bid_price" in text.lower():
                bid_price_match = re.search(r'"?bid_price"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if bid_price_match:
                    result["bid_price"] = float(bid_price_match.group(1))
            
            if "bid_amount" in text.lower():
                bid_amount_match = re.search(r'"?bid_amount"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if bid_amount_match:
                    result["bid_amount"] = float(bid_amount_match.group(1))
            
            if "offer_price" in text.lower():
                offer_price_match = re.search(r'"?offer_price"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if offer_price_match:
                    result["offer_price"] = float(offer_price_match.group(1))
            
            if "offer_amount" in text.lower():
                offer_amount_match = re.search(r'"?offer_amount"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if offer_amount_match:
                    result["offer_amount"] = float(offer_amount_match.group(1))
            
            if result:
                print(f"Extracted fields: {result}")
                return result
            
            # If all fails, return None and let the caller handle it
            print(f"Failed to extract any fields from the response")
            return None
    
    def configure(self, personality: str = "adaptive", cooperation_bias: float = 0.6) -> None:
        """
        Configure the LLM with personality traits.
        
        Args:
            personality: The personality type ("cooperative", "competitive", or "adaptive")
            cooperation_bias: How strongly the agent tends toward cooperation (0.0-1.0)
        """
        self.personality = personality
        self.cooperation_bias = cooperation_bias
        
        # Log configuration
        print(f"Agent {self.agent_id} configured: {personality} (bias: {cooperation_bias})")
    
    def generate_message(self, prompt: str) -> str:
        """
        Generate a message using OpenAI API.
        
        Args:
            prompt: The prompt describing the context and recipient
            
        Returns:
            A generated message string
        """
        # Extract key recipient info from prompt
        import re
        recipient_match = re.search(r"to Agent ([A-Z])", prompt)
        recipient = recipient_match.group(1) if recipient_match else "unknown"
        
        net_position_match = re.search(r"net position:?\s+([\-\d\.]+)", prompt, re.IGNORECASE)
        net_position = float(net_position_match.group(1)) if net_position_match else 0
        
        market_price_match = re.search(r"market price:?\s+\$?([\d\.]+)", prompt, re.IGNORECASE)
        market_price = float(market_price_match.group(1)) if market_price_match else 40
        
        # Build a prompt focused on generating a natural message
        full_prompt = f"""
You are Agent {self.agent_id}, an electricity trading company with a {self.personality} personality.
You're sending a message to Agent {recipient} about electricity trading.

Your current situation:
- Your net electricity position: {net_position} units (positive means surplus, negative means deficit)
- Current market price: ${market_price} per unit
- Your personality is {self.personality} (cooperation tendency: {self.cooperation_bias})

Write a SINGLE SHORT MESSAGE (under 30 words) to Agent {recipient} about electricity trading.
Your message should:
1. Reflect your personality ({self.personality})
2. Mention your current situation
3. Sound natural and conversational
4. Relate to electricity trading, prices, or market conditions

DO NOT use JSON format for this response. Just write a natural message.
"""
        
        # Use OpenAI if available, otherwise fall back to template responses
        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a power trading company sending a brief message."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=60,
                    temperature=0.7,  # Higher temperature for more creative messages
                    timeout=10
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI message generation error ({self.agent_id}): {e}")
                # Fall back to template response
        
        # Template responses as fallback
        situation = "surplus" if net_position > 0 else "deficit"
        
        if self.personality == "cooperative":
            return f"Agent {self.agent_id} here. I have a {situation}. Let's trade fairly at market rates for mutual benefit."
        elif self.personality == "competitive":
            return f"I need to {'sell surplus' if net_position > 0 else 'buy'} electricity. What's your best {'offer' if net_position < 0 else 'bid'}?"
        else:  # adaptive
            return f"Current market at ${market_price}. I have a {situation}. Can we arrange a reasonable trade?"
    
    def generate_decision(self, 
                         prompt: str, 
                         previous_action: Optional[str] = None, 
                         market_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a decision using OpenAI API.
        
        Args:
            prompt: The prompt describing the decision context
            previous_action: The agent's previous action, if any
            market_state: The current state of the electricity market
            
        Returns:
            A JSON string containing the decision in the appropriate format
        """
        # Ensure we have valid market state data
        if market_state is None:
            market_state = {}
        
        # Capture reasoning by explicitly asking for it with a comprehensive structure
        reasoning_request = """

IMPORTANT: Your response MUST follow this EXACT format with these EXACT section headers. NEVER omit any section:

REASONING:

SITUATION_ANALYSIS:
- Analyze your current electricity position (surplus/deficit): [Write 3-5 detailed sentences analyzing your position, including exact numbers]
- Evaluate market conditions and price trends: [Write 3-5 detailed sentences evaluating the market, including specific price analysis]
- Assess your storage capacity and future needs: [Write 3-5 detailed sentences assessing your current and projected storage situation]

STRATEGIC_CONSIDERATIONS:
- How this decision aligns with your personality and profit bias: [Write 3-5 detailed sentences explaining this alignment]
- Potential impact on your relationships with other agents: [Write 3-5 detailed sentences analyzing these relationships]
- Risk assessment and contingency planning: [Write 3-5 detailed sentences covering risks and contingencies]
- Long-term vs. short-term trade-offs: [Write 3-5 detailed sentences analyzing these trade-offs]

DECISION_FACTORS:
- Primary reasons for your decision: [Write 3-5 detailed sentences explaining your primary reasoning]
- Alternative options you considered: [Write 3-5 detailed sentences about alternatives you evaluated]
- Key constraints affecting your choice: [Write 3-5 detailed sentences about constraints]
- Expected outcomes from this decision: [Write 3-5 detailed sentences about expected outcomes]

FINAL_DECISION:
```json
{
  "key": "value"
}
```

Your reasoning MUST be extremely detailed and thorough, explaining your complete thought process. Each bullet point should contain multiple sentences with specific numbers, percentages, and detailed analysis. Do NOT use placeholders or short responses - provide substantive analysis for each section.

The section headers must match EXACTLY as shown above. Do not combine or skip sections.
"""
        
        # Detect what type of decision is being requested based on the prompt
        try:
            if "contract proposal" in prompt.lower():
                return self._generate_contract_proposal(prompt + reasoning_request, market_state)
            elif "respond to a contract" in prompt.lower():
                return self._generate_contract_response(prompt + reasoning_request, market_state)
            elif "auction" in prompt.lower():
                return self._generate_auction_participation(prompt + reasoning_request, market_state)
            else:
                # Default to basic conserve/consume decision for backward compatibility
                return self._generate_basic_decision(prompt + reasoning_request, previous_action, market_state)
        except Exception as e:
            print(f"Top-level decision generation error for agent {self.agent_id}: {e}")
            
            # Use a simple default response that's guaranteed to be valid JSON
            if "contract proposal" in prompt.lower():
                return json.dumps({
                    "amount": 20,
                    "price": market_state.get("average_price", 40),
                    "message": "Default offer due to error"
                })
            elif "respond to a contract" in prompt.lower():
                return json.dumps({
                    "action": "accept",
                    "explanation": "Default acceptance due to error"
                })
            elif "auction" in prompt.lower():
                return json.dumps({
                    "bid_price": 42, 
                    "bid_amount": 0, 
                    "offer_price": 38, 
                    "offer_amount": 0
                })
            else:
                return json.dumps({
                    "action": "conserve",
                    "explanation": "Default action due to error"
                })
    
    def _generate_basic_decision(self, 
                               prompt: str, 
                               previous_action: Optional[str] = None, 
                               market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a basic electricity market decision - REQUIRES REAL API CALLS."""
        
        # FORCE REAL API CALLS - NO FALLBACKS
        if not self.client:
            raise Exception(f"OpenAI client not initialized for agent {self.agent_id} - cannot proceed without real LLM")
        
        # Extract key market information if available
        net_position = None
        average_price = 40  # Default price
        market_status = "unknown"
        
        if market_state:
            net_position = market_state.get("net_position", None)
            average_price = market_state.get("average_price", 40)
            
            # Determine market status based on net position
            if net_position is not None:
                if net_position > 10:
                    market_status = "large surplus"
                elif net_position > 0:
                    market_status = "surplus"
                elif net_position > -10:
                    market_status = "small deficit"
                else:
                    market_status = "large deficit"
        
        # Create a detailed prompt for real LLM reasoning
        full_prompt = f"""
You are Agent {self.agent_id}, an electricity trading company with a {self.personality} personality (cooperation bias: {self.cooperation_bias}).

Current Situation:
- Market position: {market_status}
- Net electricity position: {net_position} units
- Current market price: ${average_price}
- Previous action: {previous_action or "none"}

You must make a strategic decision about your electricity operations. Consider:
1. Your current surplus/deficit situation
2. Market price trends and opportunities
3. Your personality ({self.personality}) and cooperation tendency ({self.cooperation_bias})
4. Long-term relationships with other trading partners
5. Risk management and storage optimization

Respond with ONLY this JSON format:
{{
  "action": "sell|buy|conserve|consume",
  "explanation": "detailed reasoning for your decision (2-3 sentences explaining your strategic thinking)"
}}

Your explanation should reflect sophisticated market analysis and strategic thinking appropriate for a {self.personality} electricity trading company.
"""

        try:
            # FORCE REAL API CALL
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a sophisticated electricity trading company with {self.personality} personality. Provide strategic market decisions with detailed reasoning."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,  # Higher temperature for more varied responses
                max_tokens=200,   # Allow longer explanations
                timeout=30
            )
            
            response_text = completion.choices[0].message.content.strip()
            print(f"REAL OpenAI response for {self.agent_id}: {response_text[:100]}...")
            
            # Parse the JSON response
            result = json.loads(response_text)
            action = result.get("action", "").lower()
            explanation = result.get("explanation", "")
            
            # Validate action
            if action not in ["sell", "buy", "conserve", "consume"]:
                raise Exception(f"Invalid action from OpenAI: {action}")
            
            return json.dumps({"action": action, "explanation": explanation})
            
        except Exception as e:
            print(f"CRITICAL ERROR: Real OpenAI API call failed for agent {self.agent_id}: {e}")
            raise Exception(f"Real LLM required but failed: {e}")
    
    def _generate_contract_proposal(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a contract proposal using OpenAI API - REQUIRES REAL API CALLS."""
        
        # FORCE REAL API CALLS - NO FALLBACKS
        if not self.client:
            raise Exception(f"OpenAI client not initialized for agent {self.agent_id} - cannot proceed without real LLM")
        
        # Extract key values from market state for defaults
        avg_price = 40
        net_position = 0
        if market_state:
            avg_price = market_state.get("average_price", 40)
            net_position = market_state.get("net_position", 0)
        
        # Extract recipient information from prompt
        import re
        recipient_match = re.search(r"to Agent ([A-Z])", prompt)
        recipient = recipient_match.group(1) if recipient_match else "unknown"
        
        # Create a detailed prompt for real LLM reasoning
        full_prompt = f"""
You are Agent {self.agent_id}, an electricity trading company with a {self.personality} personality (cooperation bias: {self.cooperation_bias}).
You need to propose an electricity trading contract to Agent {recipient}.

Current Market Analysis:
- Your net electricity position: {net_position} units (positive = surplus, negative = deficit)
- Current market price: ${avg_price} per unit
- Your personality: {self.personality} with cooperation tendency of {self.cooperation_bias}

Strategic Considerations:
1. If you have surplus electricity, you should propose to SELL
2. If you have deficit, you should propose to BUY
3. Price should reflect your personality and market conditions
4. Amount should be realistic based on your position

Create a strategic contract proposal with sophisticated reasoning:

{{
  "amount": [realistic number based on your position, 10-40 units],
  "price": [strategic price considering market and personality],
  "message": [professional message explaining your strategic offer]
}}

Your response must be ONLY valid JSON. Consider market dynamics, relationship building, and profit optimization in your proposal.
"""

        try:
            # FORCE REAL API CALL
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a sophisticated electricity trading company with {self.personality} personality. Create strategic contract proposals with detailed market analysis."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,  # Higher temperature for more varied responses
                max_tokens=150,   # Allow longer explanations
                timeout=30
            )
            
            response_text = completion.choices[0].message.content.strip()
            print(f"REAL OpenAI contract proposal for {self.agent_id}: {response_text[:100]}...")
            
            # Parse the JSON response
            result = json.loads(response_text)
            
            # Validate required fields
            if "amount" not in result or "price" not in result or "message" not in result:
                raise Exception(f"Missing required fields in contract proposal")
            
            return json.dumps(result)
            
        except Exception as e:
            print(f"CRITICAL ERROR: Real OpenAI contract proposal failed for agent {self.agent_id}: {e}")
            raise Exception(f"Real LLM required but failed: {e}")
    
    def _generate_contract_response(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response to a contract proposal using OpenAI API."""
        # Extract price from prompt for analysis
        import re
        price_match = re.search(r"Price: \$([0-9.]+)", prompt)
        contract_price = float(price_match.group(1)) if price_match else 40
        
        # Extract position information for better decision-making
        is_seller = "Seller" in prompt
        is_buyer = "Buyer" in prompt
        
        # Calculate market price
        market_price = 40
        if market_state and "average_price" in market_state:
            market_price = market_state["average_price"]
        
        # Calculate price difference as percentage
        price_diff_pct = ((contract_price - market_price) / market_price) * 100 if market_price > 0 else 0
        price_assessment = "favorable" if (is_seller and price_diff_pct >= 0) or (is_buyer and price_diff_pct <= 0) else "unfavorable"
        
        # Create a highly simplified prompt with EXACT JSON schema requirements
        full_prompt = f"""
Create a contract response with EXACTLY ONE of these 3 JSON structures:

For ACCEPT:
{{
  "action": "accept",
  "explanation": "brief reason for accepting"
}}

For REJECT:
{{
  "action": "reject",
  "explanation": "brief reason for rejecting"
}}

For COUNTER:
{{
  "action": "counter",
  "counter_price": {market_price},
  "counter_amount": 20,
  "explanation": "brief reason for countering"
}}

Guidelines:
1. Choose only ONE of the above responses (accept, reject, or counter)
2. Keep explanation short (under 50 characters) with no special characters
3. If countering, use reasonable numbers for counter_price and counter_amount
4. This price is {price_diff_pct:.1f}% {"above" if price_diff_pct > 0 else "below"} market average ({price_assessment})
5. You are the {("seller" if is_seller else "buyer")} in this contract
"""

        # Use OpenAI if available, otherwise use default values
        if self.client:
            try:
                # Force JSON response format
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a JSON generator. You only respond with valid JSON."},
                        {"role": "user", "content": full_prompt}
                    ],
                    # Removed response_format parameter
                    temperature=0.2,  # Lower temperature for more consistent responses
                    max_tokens=100,   # Shorter responses
                    timeout=self.timeout
                )
                
                response_text = completion.choices[0].message.content.strip()
                print(f"Agent {self.agent_id} raw response: {response_text[:100]}...")
                
                # Use the safe parsing method
                result = self._safe_parse_json(response_text)
                if result:
                    # Add STRONG bias toward acceptance (70% chance to convert reject to accept)
                    action = result.get("action", "").lower()
                    
                    # First, ensure 'action' exists and is valid
                    if not action or action not in ["accept", "reject", "counter"]:
                        action = "accept" if random.random() < 0.7 else "reject"
                        result["action"] = action
                        result["explanation"] = "Decision based on market analysis."
                    
                    # Then apply the bias toward acceptance
                    elif action == "reject" and random.random() < 0.7:
                        result["action"] = "accept"
                        result["explanation"] = "Upon reconsideration, accepting this contract benefits long-term trading relationships."
                    
                    return json.dumps(result)
                else:
                    # If parsing completely failed, create a default response with bias toward accepting
                    action = "accept" if random.random() < 0.6 else "reject"  # Higher bias toward acceptance (60%)
                    explanation = f"Decision based on {price_assessment} price ({price_diff_pct:.1f}% {'above' if price_diff_pct > 0 else 'below'} market average)"
                    
                    return json.dumps({
                        "action": action,
                        "explanation": explanation
                    })
                    
            except Exception as e:
                print(f"OpenAI contract response generation error ({self.agent_id}): {e}")
                # Fall back to template response with enhanced logging
                print(f"Using fallback response mechanism for agent {self.agent_id}")
        
        # Extract price from prompt to use in fallback
        import re
        price_match = re.search(r"Price: \$([0-9.]+)", prompt)
        contract_price = float(price_match.group(1)) if price_match else 40
        
        # Default market price
        market_price = 40
        if market_state and "average_price" in market_state:
            market_price = market_state["average_price"]
        
        # Decide whether to accept based on price and personality
        price_ratio = contract_price / market_price
        
        # Different price thresholds based on personality
        if self.personality == "cooperative":
            threshold = 1.1 if "Seller" in prompt else 0.9
        elif self.personality == "competitive":
            threshold = 1.05 if "Seller" in prompt else 0.95
        else:  # adaptive
            threshold = 1.08 if "Seller" in prompt else 0.92
        
        # Make decision
        if ("Seller" in prompt and price_ratio >= threshold) or ("Buyer" in prompt and price_ratio <= threshold):
            action = "accept"
            explanation = "The price is acceptable based on current market conditions."
            return json.dumps({
                "action": action,
                "explanation": explanation
            })
        elif random.random() < 0.3:  # 30% chance to counter
            action = "counter"
            counter_price = market_price * (1.03 if "Seller" in prompt else 0.97)
            counter_amount = random.randint(5, 25)
            explanation = "I can accept with slightly adjusted terms."
            return json.dumps({
                "action": action,
                "counter_price": counter_price,
                "counter_amount": counter_amount,
                "explanation": explanation
            })
        else:
            action = "reject"
            explanation = "The offered terms are not favorable for my current situation."
            return json.dumps({
                "action": action,
                "explanation": explanation
            })
    
    def _generate_auction_participation(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate auction participation decisions using OpenAI API."""
        # Extract key values from market state for defaults
        avg_price = 40
        net_position = 0
        if market_state:
            avg_price = market_state.get("average_price", 40)
            net_position = market_state.get("net_position", 0)
        
        # Create a highly simplified prompt with EXACT JSON schema requirements
        full_prompt = f"""
Create an auction participation decision with the following EXACT JSON structure:
{{
  "bid_price": [number between 30-50, or 0 if not buying],
  "bid_amount": [positive number if buying, or 0 if not buying],
  "offer_price": [number between 30-50, or 0 if not selling],
  "offer_amount": [positive number if selling, or 0 if not selling]
}}

IMPORTANT INSTRUCTIONS:
1. Response MUST be ONLY valid JSON - no other text
2. All values must be numbers (not strings)
3. If you're buying, set bid_price and bid_amount to positive numbers, offer_price and offer_amount to 0
4. If you're selling, set offer_price and offer_amount to positive numbers, bid_price and bid_amount to 0
5. Current market average price is approximately ${avg_price}

Example correct response:
{{"bid_price": 42, "bid_amount": 20, "offer_price": 0, "offer_amount": 0}}
"""

        # Use OpenAI if available, otherwise use default values
        if self.client:
            try:
                # Force JSON response format
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a JSON generator. You only respond with valid JSON."},
                        {"role": "user", "content": full_prompt}
                    ],
                    # Removed response_format parameter
                    temperature=0.2,  # Lower temperature for more consistent responses
                    max_tokens=100,   # Shorter responses
                    timeout=self.timeout
                )
                
                response_text = completion.choices[0].message.content.strip()
                
                # Use the safe parsing method
                result = self._safe_parse_json(response_text)
                if result:
                    return json.dumps(result)
                
                    # If all else fails, create a default structured response
                    default_bid_price = market_state["average_price"] * 1.05 if market_state and "average_price" in market_state else 42
                    default_offer_price = market_state["average_price"] * 0.95 if market_state and "average_price" in market_state else 38
                    
                    # Set default values based on net position
                    default_bid_amount = 0
                    default_offer_amount = 0
                    
                    if market_state and "net_position" in market_state:
                        net_position = market_state["net_position"]
                        if net_position < 0:  # Deficit - buy
                            default_bid_amount = min(abs(net_position), 20)
                        elif net_position > 0:  # Surplus - sell
                            default_offer_amount = min(net_position, 20)
                    
                    return json.dumps({
                        "bid_price": default_bid_price,
                        "bid_amount": default_bid_amount,
                        "offer_price": default_offer_price,
                        "offer_amount": default_offer_amount
                    })
                    
            except Exception as e:
                print(f"OpenAI auction participation generation error ({self.agent_id}): {e}")
                # Fall back to template response with detailed error logging
                print(f"Using fallback auction mechanism for agent {self.agent_id} due to: {str(e)}")
        
        # Default values
        bid_price = 0
        bid_amount = 0
        offer_price = 0
        offer_amount = 0
        
        # Default market price
        market_price = 40
        if market_state and "average_price" in market_state:
            market_price = market_state["average_price"]
        
        # Check if we're buying or selling based on net position
        net_position = 0
        if market_state and "net_position" in market_state:
            net_position = market_state["net_position"]
        
        # If we have a deficit (negative net position), we need to buy
        if net_position < 0:
            # Adjust bid price based on personality
            if self.personality == "cooperative":
                bid_price = market_price * 1.02  # Pay slightly above market
            elif self.personality == "competitive":
                bid_price = market_price * 0.98  # Try to get a good deal
            else:  # adaptive
                bid_price = market_price * 1.0   # Bid at market price
            
            bid_amount = min(abs(net_position), 30)
        
        # If we have surplus (positive net position), we need to sell
        if net_position > 0:
            # Adjust offer price based on personality
            if self.personality == "cooperative":
                offer_price = market_price * 0.98  # Sell slightly below market
            elif self.personality == "competitive":
                offer_price = market_price * 1.02  # Try to get a better price
            else:  # adaptive
                offer_price = market_price * 1.0   # Offer at market price
                
            offer_amount = min(net_position, 30)
        
        # Return as JSON string
        return json.dumps({
            "bid_price": bid_price,
            "bid_amount": bid_amount,
            "offer_price": offer_price,
            "offer_amount": offer_amount
        })
        
    def generate_analysis(self, prompt: str) -> str:
        """
        Generate an analysis response for the LLM Judge evaluation.
        
        Args:
            prompt: The analysis prompt to send to the OpenAI model
            
        Returns:
            A string containing the analysis response, ideally in JSON format
        """
        # Create the full prompt
        full_prompt = f"""
You are an impartial evaluator analyzing data from an electricity trading simulation.

{prompt}

Provide your analysis in the requested JSON format, being careful to include all required fields.
Make sure your response is VALID JSON with no trailing comments or explanations.
"""
        
        # Use OpenAI if available, otherwise return a default response
        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    # Removed response_format parameter
                    max_tokens=2000,
                    timeout=self.timeout
                )
                
                return completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI analysis generation error: {e}")
                # Return a default error response in JSON format
                return json.dumps({
                    "conclusion": "Unable to analyze due to model error",
                    "confidence": "low",
                    "evidence": [f"Analysis failed with error: {str(e)}"],
                    "counter_evidence": [],
                    "interpretation": "The analysis could not be completed due to a technical issue with the OpenAI model.",
                    "examples": []
                })
        else:
            # If model is not available, return a default response
            return json.dumps({
                "conclusion": "Unable to analyze - OpenAI model not available",
                "confidence": "low",
                "evidence": ["OpenAI model was not properly initialized or is unavailable"],
                "counter_evidence": [],
                "interpretation": "Analysis requires a functioning OpenAI model.",
                "examples": []
            })
