using Documenter
using NGCLearn

DocMeta.setdocmeta!(NGCLearn, :DocTestSetup, :(using NGCLearn); recursive=true)

makedocs(;
    modules=[NGCLearn],
    authors="CognitiveSubstrates AI",
    repo=Remotes.GitHub("CognitiveSubstratesAI", "NGCLearn"),
    sitename="NGCLearn.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cognitivesubstratesai.github.io/NGCLearn/stable/",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Quickstart" => "getting_started/quickstart.md"
        ],
        "Components" => "components.md",
        "Models" => "models.md",
        "Architecture & Design" => "architecture.md",
        "API Reference" => "api/index.md"
    ],
    warnonly=[:missing_docs, :cross_references]
)

deploydocs(;
    repo="github.com/CognitiveSubstratesAI/NGCLearn",
    devbranch="main"
)
