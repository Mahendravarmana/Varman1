import win32evtlog
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#############################################
# CONFIGURATION (INSIDE PYTHON)
#############################################

CONFIG = {
    "log_sources": {
        "system_log": "System",
        "application_log": "Setup",
        "security_log": "Security"
    }
}

log_sources = CONFIG["log_sources"]

#############################################
# READ WINDOWS EVENT LOGS
#############################################

def read_windows_logs(log_sources):

    server = "localhost"
    messages = []

    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ

    for name, logtype in log_sources.items():

        print(f"Reading {logtype}")

        try:
            handle = win32evtlog.OpenEventLog(server, logtype)

            while True:

                events = win32evtlog.ReadEventLog(handle, flags, 0)

                if not events:
                    break

                for event in events:

                    if event.StringInserts:
                        msg = " ".join(event.StringInserts)
                        messages.append(msg)

        except Exception as e:
            print(f"Error reading {logtype}: {e}")

    return messages


logs = read_windows_logs(log_sources)

print("Total logs collected:", len(logs))

#############################################
# DATA PREPARATION
#############################################
if len(logs) == 0:
    raise RuntimeError("No logs were collected from Windows Event Log")
labels = [0 if "error" not in log.lower() else 1 for log in logs]
print(f" reading  {logs[:5]}")
vectorizer = CountVectorizer(max_features=2000)

X = vectorizer.fit_transform(logs).toarray()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#############################################
# TRANSFORMER MODEL
#############################################

class LogTransformer(nn.Module):

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.classifier(x)


#############################################
# TRAIN MODEL
#############################################

input_dim = X.shape[1]

model = LogTransformer(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#############################################
# TEST MODEL
#############################################

with torch.no_grad():

    preds = model(X_test)

    predicted = torch.argmax(preds, dim=1)

    accuracy = (predicted == y_test).float().mean()

    print("Test Accuracy:", accuracy.item())