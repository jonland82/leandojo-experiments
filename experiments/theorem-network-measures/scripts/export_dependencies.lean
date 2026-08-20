import Mathlib
import Lean.Util.FoldConsts

open Lean Elab Command

/-!
Emit the direct kernel-level constant dependencies of every declaration loaded
by `import Mathlib`. Output is tab-separated: declaration, then dependencies.
Names cannot contain tab characters, so the format is lossless and easy to
stream from Python.
-/

run_cmd do
  let env ← getEnv
  env.constants.forM fun name info => do
    let mut line := toString name
    for dependency in info.getUsedConstantsAsSet do
      if dependency != name then
        line := line ++ "\t" ++ toString dependency
    IO.println line
