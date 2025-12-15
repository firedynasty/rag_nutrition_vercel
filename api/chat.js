// Vercel Serverless Function for OpenAI Chat
// Environment variables (set in Vercel dashboard):
// - OPENAI_API_KEY: Your OpenAI API key (server-side only, NOT exposed to client)
// - ACCESS_CODE: Password users must enter to use Stanley's key

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { messages, model, accessCode, userApiKey, webSearch } = req.body;

  // Determine which API key to use
  let apiKey;

  if (userApiKey) {
    // User provided their own API key
    apiKey = userApiKey;
  } else if (accessCode) {
    // User wants to use Stanley's key - validate access code
    const validAccessCode = process.env.ACCESS_CODE;
    if (accessCode !== validAccessCode) {
      return res.status(401).json({ error: 'Invalid access code' });
    }
    apiKey = process.env.OPENAI_API_KEY;
  } else {
    return res.status(400).json({ error: 'No API key or access code provided' });
  }

  if (!apiKey) {
    return res.status(500).json({ error: 'API key not configured' });
  }

  try {
    let response;
    let assistantContent;

    if (webSearch) {
      // Use Responses API with web_search tool
      response = await fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: model || 'gpt-4o-mini',
          tools: [{ type: 'web_search' }],
          tool_choice: 'auto',
          input: messages[messages.length - 1].content,
        }),
      });

      const responseText = await response.text();
      console.log('OpenAI web search response status:', response.status);

      if (!response.ok) {
        let errorMessage = 'OpenAI API request failed';
        try {
          const errorData = JSON.parse(responseText);
          errorMessage = errorData.error?.message || errorMessage;
        } catch (e) {
          errorMessage = responseText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = JSON.parse(responseText);

      // Extract text from response - output_text is the direct property
      assistantContent = data.output_text;
      if (!assistantContent && data.output) {
        for (const item of data.output) {
          if (item.type === 'message' && item.content) {
            for (const content of item.content) {
              if (content.type === 'output_text' && content.text) {
                assistantContent = content.text;
                break;
              }
            }
          }
          if (assistantContent) break;
        }
      }

      if (!assistantContent) {
        throw new Error('No response from web search');
      }
    } else {
      // Use standard Chat Completions API
      response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: model || 'gpt-4o-mini',
          max_tokens: 4096,
          messages: messages,
        }),
      });

      const responseText = await response.text();
      console.log('OpenAI response status:', response.status);

      if (!response.ok) {
        let errorMessage = 'OpenAI API request failed';
        try {
          const errorData = JSON.parse(responseText);
          errorMessage = errorData.error?.message || errorMessage;
        } catch (e) {
          errorMessage = responseText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = JSON.parse(responseText);
      if (!data.choices || !data.choices[0] || !data.choices[0].message) {
        console.error('Unexpected response structure:', data);
        throw new Error('Unexpected response from OpenAI');
      }
      assistantContent = data.choices[0].message.content;
    }

    return res.status(200).json({ content: assistantContent });
  } catch (error) {
    console.error('OpenAI API error:', error);
    return res.status(500).json({ error: error.message || 'Failed to get response from OpenAI' });
  }
}
