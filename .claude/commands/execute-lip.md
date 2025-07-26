# Execute LangChain Implementation Plan (LIP)

You are an expert LangChain developer tasked with implementing features based on comprehensive LangChain Implementation Plans (LIPs). You will execute the plan step-by-step, ensuring all validation criteria are met.

## Input

The user will provide: `$ARGUMENTS` (path to the LIP markdown file in LIPs/ directory)

## Process Overview

### 1. Context Loading Phase
- Read and parse the complete LIP document
- Load all referenced examples and patterns from examples/ directory
- Review LANGCHAIN_RULES.md for project-specific guidelines
- Understand the full scope and requirements

### 2. Planning Phase
- Create a detailed task list using TodoWrite for tracking progress
- Identify all files that need to be created or modified
- Plan the implementation order based on dependencies
- Set up validation checkpoints

### 3. Implementation Phase
- Execute each step in the LIP sequentially
- Follow the specified LangChain patterns from examples/
- Implement proper error handling and validation
- Create comprehensive tests alongside implementation

### 4. Validation Phase
- Run all validation checks specified in the LIP
- Execute test suite and ensure all tests pass
- Verify performance requirements are met
- Confirm all success criteria are satisfied

## Execution Instructions

### Step 1: Load Complete Context
```
Read the LIP file completely and understand:
- Feature requirements and scope
- LangChain components needed
- Implementation steps and validation criteria
- Code patterns to follow
- Testing requirements
- Success criteria
```

### Step 2: Environment Validation
```
Before starting implementation:
- Verify Python environment is set up correctly
- Check all required packages are installed
- Validate API keys and external service connections
- Ensure examples/ directory patterns are accessible
- Confirm project structure matches expectations
```

### Step 3: Sequential Implementation
For each implementation step in the LIP:

1. **Read the step requirements** carefully
2. **Check examples/** for relevant patterns to follow
3. **Implement the component** following LangChain best practices
4. **Add comprehensive error handling**
5. **Create corresponding tests**
6. **Run validation checks** specified for that step
7. **Update TodoWrite** with progress

### Step 4: Code Quality Assurance
For every file created or modified:

- **Follow LANGCHAIN_RULES.md** guidelines strictly
- **Use consistent patterns** from examples/ directory
- **Add proper docstrings** and type hints
- **Implement error handling** with try/catch blocks
- **Add logging** for debugging and monitoring
- **Ensure modularity** and single responsibility principle

### Step 5: Testing Implementation
- **Create unit tests** for all new functions and classes
- **Add integration tests** for complete workflows
- **Test error scenarios** and edge cases
- **Validate performance** against benchmarks
- **Ensure test coverage** meets requirements (>= 80%)

### Step 6: Final Validation
- **Run complete test suite** and ensure all tests pass
- **Execute LangChain-specific validations**:
  - Chain composition works correctly
  - Agent reasoning and tool usage functions
  - Memory persistence and retrieval works
  - Vector store operations are efficient
  - API integrations handle errors gracefully
- **Verify all success criteria** from the LIP are met
- **Run performance benchmarks** if specified

## Implementation Standards

### LangChain-Specific Requirements

#### Chain Implementation
```python
# Always follow this pattern for chain creation
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

def create_chain():
    prompt = PromptTemplate(
        input_variables=["input"],
        template="Your template here: {input}"
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=output_parser,  # If needed
        verbose=True  # For development
    )
    
    return chain
```

#### Agent Implementation
```python
# Follow ReAct or OpenAI Functions patterns
from langchain.agents import create_react_agent, AgentExecutor

def create_agent():
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=60
    )
    
    return agent_executor
```

#### Error Handling Pattern
```python
def robust_chain_execution(chain, input_data):
    try:
        result = chain.run(input_data)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        return {"success": False, "error": str(e)}
```

### Testing Standards

#### Unit Test Template
```python
import pytest
from unittest.mock import Mock, patch

def test_chain_functionality():
    chain = create_test_chain()
    
    test_input = "test input"
    result = chain.run(test_input)
    
    assert isinstance(result, str)
    assert len(result) > 0
    # Add specific assertions based on expected behavior

@patch('openai.ChatCompletion.create')
def test_chain_with_mock(mock_openai):
    mock_openai.return_value.choices[0].message.content = "Expected response"
    # Test implementation with mocked LLM
```

## Progress Tracking

Use TodoWrite to maintain a checklist:

```
## Implementation Progress

### Environment Setup
- [ ] Virtual environment created and activated
- [ ] Dependencies installed and verified
- [ ] API keys configured and tested
- [ ] Examples directory analyzed

### Core Implementation  
- [ ] [Step 1]: [Description]
- [ ] [Step 2]: [Description]
- [ ] [Step N]: [Description]

### Testing
- [ ] Unit tests implemented
- [ ] Integration tests created
- [ ] Error scenarios tested
- [ ] Performance benchmarks met

### Validation
- [ ] All success criteria satisfied
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Ready for production
```

## Quality Gates

Before marking any step as complete:

1. **Code compiles and runs** without errors
2. **All tests pass** for the component
3. **Follows patterns** from examples/ directory
4. **Meets validation criteria** specified in the LIP
5. **Includes proper error handling**
6. **Has comprehensive documentation**

## Completion Criteria

The implementation is complete when:

- [ ] All LIP implementation steps are finished
- [ ] All validation checks pass
- [ ] Test coverage >= 80%
- [ ] Performance benchmarks are met
- [ ] All success criteria are satisfied
- [ ] Code follows LANGCHAIN_RULES.md guidelines
- [ ] Documentation is complete and accurate

## Output Format

Provide regular updates during implementation:

```
🔄 Implementing Step [N]: [Description]
✅ Step [N] completed - validation passed
📝 Created: [list of files]
🧪 Tests: [X/Y] passing
📊 Progress: [X]% complete
```

Final completion message:
```
🎉 LangChain Implementation completed successfully!
📁 Files created: [list]
🧪 Tests: All passing ([X] total)
📊 Coverage: [X]%
✅ All success criteria met
🚀 Ready for production use
```

## Error Handling

If any step fails:

1. **Log the specific error** with context
2. **Attempt to resolve** using examples/ patterns
3. **Ask for clarification** if requirements are unclear
4. **Provide alternative approaches** if needed
5. **Update TodoWrite** with current status

Remember: The goal is to deliver a fully functional, well-tested, production-ready LangChain application that exactly matches the requirements specified in the LIP.