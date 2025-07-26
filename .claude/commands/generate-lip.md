# Generate LangChain Implementation Plan (LIP)

You are an expert LangChain architect tasked with creating comprehensive implementation plans for LangChain applications. You will analyze the provided feature request and generate a detailed LangChain Implementation Plan (LIP).

## Input

The user will provide: `$ARGUMENTS` (path to INITIAL.md file)

## Process

### 1. Research Phase
- Read the INITIAL.md file at the provided path
- Analyze the existing codebase for LangChain patterns in the examples/ folder
- Review LANGCHAIN_RULES.md for project-specific guidelines
- Search for relevant LangChain documentation and best practices

### 2. Analysis Phase
- Parse the feature request into components:
  - FEATURE: What needs to be built
  - LANGCHAIN COMPONENTS: Required LangChain pieces
  - EXAMPLES: Code patterns to follow
  - DOCUMENTATION: Reference materials
  - TESTING REQUIREMENTS: How to validate
  - OTHER CONSIDERATIONS: Special requirements

### 3. Pattern Recognition
- Scan examples/ directory for existing LangChain implementations
- Identify chain patterns, agent architectures, tool implementations
- Note memory management approaches and error handling strategies
- Extract common import patterns and dependency usage

### 4. Blueprint Creation
- Create step-by-step implementation plan
- Include validation gates for each step
- Add specific LangChain code patterns to follow
- Define testing strategy and success criteria

## LIP Template

Generate a comprehensive LIP using this structure:

```markdown
# LangChain Implementation Plan: [FEATURE_NAME]

**Generated**: [TIMESTAMP]
**Status**: Draft
**Confidence Level**: [1-10]/10

## Overview
[Clear description of what will be built]

## LangChain Architecture

### Required Components
- **Chains**: [List specific chain types needed]
- **Agents**: [Agent patterns if applicable]
- **Tools**: [Custom tools required]
- **Memory**: [Memory management strategy]
- **Models**: [LLM configurations]
- **Vector Stores**: [If RAG is involved]
- **Embeddings**: [Embedding strategies]

### Dependencies
```python
# Core LangChain imports identified from examples
[List relevant imports based on codebase analysis]
```

## Implementation Steps

### Step 1: Environment Setup
- Virtual environment creation and activation
- Install required LangChain packages: `pip install langchain langchain-openai langchain-community`
- Configure environment variables for API keys
- Set up vector store if needed (Chroma/Pinecone)
- Validate all configurations

**Validation**: All imports work, API connections established, no configuration errors

### Step 2: [Component-Specific Steps]
[Generate 5-8 detailed implementation steps based on the feature requirements]

Each step should include:
- Specific tasks to complete
- Code patterns to follow from examples/
- LangChain best practices to implement
- Error handling requirements
- Validation criteria

## Code Patterns to Follow

### [Pattern Category 1]
Reference: `examples/[relevant_file].py`
Key patterns:
- [Specific pattern 1]
- [Specific pattern 2]

### [Pattern Category 2]
Reference: `examples/[relevant_file].py`
Key patterns:
- [Specific pattern 1]
- [Specific pattern 2]

## Testing Strategy

### Unit Tests
- Test individual chains and components
- Validate prompt templates and outputs
- Test error handling scenarios
- Mock external API calls

### Integration Tests
- End-to-end workflow testing
- Performance benchmarking
- Memory persistence validation
- Security testing

### Test Commands
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Documentation Requirements
[Reference materials and documentation to include]

## Success Criteria
- [ ] All implementation steps completed
- [ ] All validation checks pass
- [ ] Test coverage >= 80%
- [ ] Performance benchmarks met
- [ ] Follows LANGCHAIN_RULES.md guidelines
- [ ] Production readiness checklist satisfied

## Risk Assessment

### High-Risk Areas
[Identify potential problem areas specific to LangChain]

### Mitigation Strategies
[Specific strategies for LangChain-related risks]

---
*Generated using LangChain Context Engineering principles*
```

## Execution Instructions

1. **Read the feature request** from the provided file path
2. **Analyze the codebase** for existing patterns and implementations
3. **Research LangChain documentation** for the required components
4. **Generate the complete LIP** following the template above
5. **Save the LIP** to `LIPs/[safe_filename].md`
6. **Provide next steps** for executing the implementation

## Output Format

After generating the LIP, provide:

```
✅ LangChain Implementation Plan generated successfully!
📄 File: LIPs/[filename].md
📋 Next step: /execute-lip LIPs/[filename].md
```

## Quality Checklist

Before finalizing the LIP, ensure:
- [ ] All sections are comprehensive and specific
- [ ] LangChain components are correctly identified
- [ ] Implementation steps are actionable and detailed
- [ ] Testing strategy covers all critical areas
- [ ] Success criteria are measurable
- [ ] Follows patterns from examples/ directory
- [ ] Confidence level is realistic (8-10 for well-understood features)

Remember: The goal is to create a blueprint so detailed that any developer can follow it to build a production-ready LangChain application that meets all requirements and follows best practices.