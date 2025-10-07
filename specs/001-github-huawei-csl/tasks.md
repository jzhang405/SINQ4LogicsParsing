# Tasks: SINQ Quantization for Logics-Parsing Model

**Input**: Design documents from `/specs/001-github-huawei-csl/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize Python project with PyTorch, Transformers, SINQ dependencies
- [ ] T003 [P] Configure linting and formatting tools

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Contract test model loading in tests/contract/test_model_loading.py
- [ ] T005 [P] Contract test quantization process in tests/contract/test_quantization.py
- [ ] T006 [P] Integration test model loading from cache in tests/integration/test_model_loading.py
- [ ] T007 [P] Integration test quantization pipeline in tests/integration/test_quantization_pipeline.py
- [ ] T008 [P] Integration test model validation in tests/integration/test_validation.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T009 [P] SINQ quantization implementation in src/models/quantization.py
- [ ] T010 [P] Model validation logic in src/models/validation.py
- [ ] T011 [P] Model loader service in src/services/model_loader.py
- [ ] T012 [P] Quantizer service in src/services/quantizer.py
- [ ] T013 CLI main interface in src/cli/main.py
- [ ] T014 Configuration management in src/utils/config.py
- [ ] T015 Logging setup in src/utils/logging.py

## Phase 3.4: Integration
- [ ] T016 Main quantization script in scripts/quantize_model.py
- [ ] T017 Model validation script in scripts/validate_model.py
- [ ] T018 Environment setup script in scripts/setup.sh
- [ ] T019 Default quantization parameters in config/default.yaml
- [ ] T020 Logics-Parsing specific config in config/logics_parsing.yaml

## Phase 3.5: Polish
- [ ] T021 [P] Unit tests for quantization in tests/unit/test_quantization.py
- [ ] T022 [P] Unit tests for validation in tests/unit/test_validation.py
- [ ] T023 Performance tests for model size reduction and accuracy
- [ ] T024 [P] Update documentation and README
- [ ] T025 Remove duplication and optimize code
- [ ] T026 Run comprehensive validation tests

## Dependencies
- Tests (T004-T008) before implementation (T009-T015)
- T009 blocks T012 (quantization implementation before quantizer service)
- T010 blocks T011 (validation before model loader)
- Implementation (T009-T020) before polish (T021-T026)

## Parallel Example
```
# Launch T004-T008 together:
Task: "Contract test model loading in tests/contract/test_model_loading.py"
Task: "Contract test quantization process in tests/contract/test_quantization.py"
Task: "Integration test model loading from cache in tests/integration/test_model_loading.py"
Task: "Integration test quantization pipeline in tests/integration/test_quantization_pipeline.py"
Task: "Integration test model validation in tests/integration/test_validation.py"

# Launch T009-T012 together:
Task: "SINQ quantization implementation in src/models/quantization.py"
Task: "Model validation logic in src/models/validation.py"
Task: "Model loader service in src/services/model_loader.py"
Task: "Quantizer service in src/services/quantizer.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Model location: /Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/
- Performance targets: Model size reduction >50%, accuracy preservation >95%

## Task Generation Rules
*Applied during main() execution*

1. **From Project Structure**:
   - Each module file → implementation task [P]
   - Each test file → test task [P]
   - Each script → integration task
   
2. **From Requirements**:
   - Model loading → model loader task
   - Quantization → quantization service task
   - Validation → validation logic task
   - CLI interface → CLI implementation task

3. **Ordering**:
   - Setup → Tests → Models → Services → Scripts → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [ ] All core modules have implementation tasks
- [ ] All test categories have test tasks
- [ ] All tests come before implementation
- [ ] Parallel tasks truly independent
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task