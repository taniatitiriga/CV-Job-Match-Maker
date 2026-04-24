// Defining a data model.
class Resume {
  title string
  positions Position[]
  skills string[]
}

class Position {
  job_title string
  company string
  start_date string
  end_date string
  description string[]
}

// Create a function to extract the resume from a string.
function ExtractResume(resume: string) -> Resume {
  // Specify a client as provider/model-name
  // You can also use custom LLM params with a custom client name from clients.baml like "client CustomGPT5" or "client CustomSonnet4"
  client "CustomOllama" // Set OLLAMA_HOST to use this client.
  prompt #"
    Extract from this content:
    {{ resume }}

    {{ ctx.output_format }}
  "#
}

function ExtractResumeLlamaCpp(resume: string) -> Resume {
  client "LlamaCpp"
  prompt #"
    Extract from this content:
    {{ resume }}

    {{ ctx.output_format }}
  "#
}

function ExtractResumeLlamaCppPC(resume: string) -> Resume {
  client "LlamaCppPCphi4Mini"
  prompt #"
    Extract from this content:
    {{ resume }}

    {{ ctx.output_format }}
  "#
}
// Test the function with a sample resume. Open the VSCode playground to run this.
test vaibhav_resume {
  functions [ExtractResume, ExtractResumeLlamaCpp, ExtractResumeLlamaCppPC]
  args {
    resume #"
      Vaibhav Gupta
      vbv@boundaryml.com

      Experience:
      - Title: Founder
        Company: BoundaryML
        Start Date: 2022-01-01
        End Date: Present
        Description:
          - Founded and currently leading BoundaryML, an AI/ML startup.
          - Developed computer vision models and deployed scalable ML systems.
          - Built and optimized vision pipelines for enterprise products.
          - Built and optimized vision pipelines for enterprise products.
          - Built and optimized vision pipelines for enterprise products.
          - Built and optimized vision pipelines for enterprise products.
      - Title: CV Engineer
        Company: Google
        Start Date: 2020-06-01
        End Date: 2022-01-01
        Description:
          - Developed computer vision models and deployed scalable ML systems.
      - Title: CV Engineer
        Company: Microsoft
        Start Date: 2018-01-01
        End Date: 2020-06-01
        Description:
          - Built and optimized vision pipelines for enterprise products.

      Skills:
      - Rust
      - C++
 
    "#
  }
}