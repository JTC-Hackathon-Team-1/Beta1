#!/bin/bash
# scripts/pretty_curl.sh
# Makes CasaLingua API calls with prettified output

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
WHITE='\033[0;97m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
RESET='\033[0m'

# Function to print a fancy header
print_header() {
    local text=$1
    echo -e "\n${BOLD}${BLUE}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
    echo -e "┃ ${YELLOW}${text}${BLUE}${BOLD} ${BLUE}┃"
    echo -e "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${RESET}\n"
}

# Function to print a section header
print_section() {
    local text=$1
    echo -e "\n${BOLD}${MAGENTA}┌──────────────────────────────────────────────────────────────┐"
    echo -e "│ ${text}${BOLD}${MAGENTA} │"
    echo -e "└──────────────────────────────────────────────────────────────┘${RESET}\n"
}

# Function to print a key-value pair
print_kv() {
    local key=$1
    local value=$2
    echo -e "${BOLD}${CYAN}${key}:${RESET} ${value}"
}

# Function to print formatted JSON
print_formatted_json() {
    local json=$1
    echo $json | python3 -m json.tool | sed -E \
        -e "s/\"([^\"]*)\":/\"${CYAN}\1${RESET}\":/" \
        -e "s/: \"([^\"]*)\"/: \"${GREEN}\1${RESET}\"/" \
        -e "s/: ([0-9]+\.[0-9]+)/: ${YELLOW}\1${RESET}/" \
        -e "s/: ([0-9]+)/: ${YELLOW}\1${RESET}/" \
        -e "s/: (true|false)/: ${MAGENTA}\1${RESET}/" \
        -e "s/: null/: ${RED}null${RESET}/"
}

# Function to format a number with 2 decimal places
format_decimal() {
    local num=$1
    printf "%.2f" $num
}

# Server URL (default to localhost)
API_URL=${API_URL:-"http://localhost:8000"}

# Check if URL is passed as argument
if [ $# -ge 1 ]; then
    API_URL=$1
fi

# Print welcome message
print_header "CasaLingua API Tester"
echo -e "${GRAY}This tool makes test requests to the CasaLingua API with pretty output${RESET}"
echo -e "${GRAY}Server: ${UNDERLINE}${API_URL}${RESET}\n"

# Function to make a request and display results
make_request() {
    local name=$1
    local source_lang=$2
    local target_lang=$3
    local text=$4
    
    # Generate a unique session ID
    local session_id="test-$(date +%s)"
    
    print_section "Request: ${name}"
    print_kv "Session ID" "${session_id}"
    print_kv "Source Language" "${YELLOW}${source_lang}${RESET}"
    print_kv "Target Language" "${YELLOW}${target_lang}${RESET}"
    print_kv "Text" "\"${GRAY}${text}${RESET}\""
    
    echo -e "\n${GRAY}Sending request...${RESET}"
    
    # Make the API request
    local start_time=$(date +%s.%N)
    local response=$(curl -s -X POST "${API_URL}/pipeline" \
        -H "Content-Type: application/json" \
        -d "{
            \"session_id\": \"${session_id}\",
            \"source_language\": \"${source_lang}\",
            \"target_language\": \"${target_lang}\",
            \"text\": \"${text}\"
        }")
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    local elapsed_formatted=$(printf "%.2f" $elapsed)
    
    # Check if response is valid JSON
    if echo "$response" | jq -e . >/dev/null 2>&1; then
        print_section "Response (${YELLOW}${elapsed_formatted}s${MAGENTA})"
        
        # Extract the translation
        local translation=$(echo "$response" | jq -r '.translation')
        echo -e "${BOLD}${GREEN}Translation:${RESET}"
        echo -e "${BLUE}\"${WHITE}${translation}${BLUE}\"${RESET}\n"
        
        # Extract entities
        local entities=$(echo "$response" | jq -r '.entities')
        local entity_count=$(echo "$entities" | jq -r 'length')
        
        echo -e "${BOLD}${GREEN}Entities (${entity_count}):${RESET}"
        if [ "$entity_count" != "0" ]; then
            echo "$response" | jq -r '.entities[] | "\(.text) (\(.label))"' | \
            while read entity; do
                local ent_text=$(echo "$entity" | cut -d'(' -f1 | sed 's/ *$//')
                local ent_label=$(echo "$entity" | cut -d'(' -f2 | sed 's/)//')
                echo -e "  • ${CYAN}${ent_text}${RESET} (${MAGENTA}${ent_label}${RESET})"
            done
        else
            echo -e "  ${GRAY}No entities found${RESET}"
        fi
        echo ""
        
        # Extract scores
        local veracity=$(echo "$response" | jq -r '.veracity_score')
        local bias=$(echo "$response" | jq -r '.bias_score')
        
        # Format scores to 2 decimal places
        local veracity_formatted=$(printf "%.2f" $veracity)
        local bias_formatted=$(printf "%.2f" $bias)
        
        # Color code the scores
        local veracity_color=$RED
        if (( $(echo "$veracity > 0.7" | bc -l) )); then
            veracity_color=$GREEN
        elif (( $(echo "$veracity > 0.4" | bc -l) )); then
            veracity_color=$YELLOW
        fi
        
        local bias_color=$GREEN
        if (( $(echo "$bias > 0.6" | bc -l) )); then
            bias_color=$RED
        elif (( $(echo "$bias > 0.3" | bc -l) )); then
            bias_color=$YELLOW
        fi
        
        echo -e "${BOLD}${GREEN}Scores:${RESET}"
        echo -e "  • Veracity: ${veracity_color}${veracity_formatted}${RESET}"
        echo -e "  • Bias:     ${bias_color}${bias_formatted}${RESET}"
        echo ""
        
        # Extract audit log
        echo -e "${BOLD}${GREEN}Audit Log:${RESET}"
        echo "$response" | jq '.audit_log' | \
        grep -v "timestamp" | \
        sed -E \
            -e "s/\"([^\"]*)\":/  • ${CYAN}\1${RESET}:/" \
            -e "s/: ([0-9]+\.[0-9]+)/: ${YELLOW}\1${RESET}/" \
            -e "s/: ([0-9]+)/: ${YELLOW}\1${RESET}/"
        
        echo -e "\n${GRAY}Complete JSON response:${RESET}"
        print_formatted_json "$response"
    else
        # There was an error
        print_section "Error"
        echo -e "${RED}${response}${RESET}"
    fi
    
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

# Test case templates
test_legalese() {
    make_request "Legalese to Plain English" "legalese" "plain_english" "Notwithstanding anything to the contrary herein, this Agreement shall commence on the Effective Date and shall continue in full force and effect until terminated in accordance with Section 9 below."
}

test_complex_english() {
    make_request "Complex English Simplification" "en" "plain_english" "The implementation of quantum algorithms necessitates the utilization of specialized hardware architectures capable of maintaining coherent quantum states for durations sufficient to execute computational procedures of significant complexity."
}

test_entities() {
    make_request "Entity Recognition" "en" "plain_english" "Apple CEO Tim Cook announced new products at their headquarters in Cupertino, California. The event took place on Tuesday and featured the release of the iPhone 15."
}

test_spanish() {
    make_request "Spanish to English" "es" "en" "El gato negro se sienta en la ventana y mira las estrellas."
}

# Check for specific test case argument
if [ "$2" == "legalese" ]; then
    test_legalese
elif [ "$2" == "complex" ]; then
    test_complex_english
elif [ "$2" == "entities" ]; then
    test_entities
elif [ "$2" == "spanish" ]; then
    test_spanish
else
    # Run all tests
    test_legalese
    test_complex_english
    test_entities
    test_spanish
fi

echo -e "${GREEN}All tests completed!${RESET}"