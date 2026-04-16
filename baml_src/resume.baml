// Defining a data model.
class Resume {
  name string
  email string
  experience string[]
  skills string[]
}

// Create a function to extract the resume from a string.
function ExtractResume(resume: string) -> Resume {
  // Specify a client as provider/model-name
  // You can also use custom LLM params with a custom client name from clients.baml like "client CustomGPT5" or "client CustomSonnet4"
  client "ollama/granite4" // Set OLLAMA_HOST to use this client.
  prompt #"
    Extract from this content:
    {{ resume }}

    {{ ctx.output_format }}
  "#
}



// Test the function with a sample resume. Open the VSCode playground to run this.
test vaibhav_resume {
  functions [ExtractResume]
  args {
    resume #"
      Vaibhav Gupta
      vbv@boundaryml.com

      Experience:
      - Founder at BoundaryML
      - CV Engineer at Google
      - CV Engineer at Microsoft

      Skills:
      - Rust
      - C++
    "#
  }
}

test vaibhav_resume_2 {
  functions [ExtractResume]
  args {
    resume #"
      Vaibhav Gupta
      vbv@boundaryml.com

      Experience:
      - Founder at BoundaryML
      - CV Engineer at Google
      - CV Engineer at Microsoft
      - CV Engineer at Apple
      - CV Engineer at Amazon
      - CV Engineer at Meta
      - CV Engineer at Twitter
      - CV Engineer at LinkedIn
      - CV Engineer at Instagram
      - CV Engineer at TikTok
      - CV Engineer at YouTube
      - CV Engineer at Snapchat
      - CV Engineer at Pinterest
      - CV Engineer at Reddit
      - CV Engineer at Discord
      - CV Engineer at Slack
      - CV Engineer at Zoom
      - CV Engineer at Microsoft Teams
      - CV Engineer at Microsoft Office
      - CV Engineer at Microsoft Excel

      Skills:
      - Rust
      - C++
      - Python
      - Java
      - JavaScript
      - TypeScript
      - HTML
      - CSS
      - SQL
      - NoSQL
      - Docker
      - Kubernetes
      - Terraform
      - Ansible
      - Puppet
      - Chef
      - Salt
      - Jenkins
      - Git
      - GitHub
      - GitLab
      - Bitbucket
      - Jira
      - Confluence
      - Slack
      - Zoom
      - Microsoft Teams
      - Microsoft Office
      - Microsoft Excel
      - Microsoft PowerPoint
      - Microsoft Word
      - Microsoft Outlook
      - Microsoft Teams
      - Microsoft Teams
    "#
  }
}