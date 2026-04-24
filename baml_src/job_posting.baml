// Defining a data model for a job posting.

enum JobPostingDomain {
  ACCOUNTANT
  ADVOCATE
  AGRICULTURE
  APPAREL
  ARTS
  AUTOMOBILE
  AVIATION
  BANKING
  BPO
  BUSINESS_DEVELOPMENT @alias("BUSINESS-DEVELOPMENT")
  CHEF
  CONSTRUCTION
  CONSULTANT
  DESIGNER
  DIGITAL_MEDIA @alias("DIGITAL-MEDIA")
  ENGINEERING
  FINANCE
  FITNESS
  HEALTHCARE
  HR
  INFORMATION_TECHNOLOGY @alias("INFORMATION-TECHNOLOGY")
  PUBLIC_RELATIONS @alias("PUBLIC-RELATIONS")
  SALES
  TEACHER
}

class JobPosting {
  job_title string
  domain JobPostingDomain
  company string
  description string
  responsibilities string[]
  required_qualifications string[]
}

function ExtractJobPostingLlamaCppPC(job_posting: string) -> JobPosting {
  client "LlamaCppPCphi4Mini"
  prompt #"
    Extract from this content:
    {{ job_posting }}

    {{ ctx.output_format }}
  "#
}

test job_posting {
  functions [ExtractJobPostingLlamaCppPC]
  args {
    job_posting #"
      Job Title: Software Engineer
      Company: Google
      Description: We are looking for a software engineer with 3 years of experience in building web applications.
      Responsibilities:
      - Build web applications
      - Develop web applications
      - Deploy web applications
      - Maintain web applications
      - Develop web applications
      Required Qualifications:
      - Bachelor's degree in Computer Science or related field
      - 3 years of experience in building web applications
      - Strong knowledge of web development frameworks
    "#
  }
}