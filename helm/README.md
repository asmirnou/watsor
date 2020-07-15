# Watsor Helm Chart

This is Helm chart to deploy [Watsor](https://github.com/asmirnou/watsor) on Kubernetes.

## Quick Start

To install the Watsor chart with default options:

```bash
helm repo add asmirnou https://asmirnou.github.io/watsor-helm-chart
helm repo update
helm install asmirnou/watsor
```

## Installation

1. Customize the values.yaml for your needs.

1. Deploy with helm:

    ```bash
    helm install \
      --name your-release \
      --values your-values.yaml \
      asmirnou/watsor
    ```

## Configuration

See [values.yaml](values.yaml) for details.

|              Parameter       |                    Description                             |                     Default                      |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| `replicaCount`               | Number of replicas                                         | `1`                                              |
| `strategy`                   | The strategy used to replace old Pods by new ones          | `Recreate`                                       |
| `image.repository`           | Container image repository and name                        | `smirnou/watsor`                                 |
| `image.tag`                  | Container image tag                                        | `""`                                             |
| `image.pullPolicy`           | Container image pull policy                                | `IfNotPresent`                                   |
| `imagePullSecrets`           | The list of secrets with the repository credentials        | `[]`                                             |
| `nameOverride`               | Name Override                                              | `""`                                             |
| `fullnameOverride`           | FullName Override                                          | `""`                                             |
| `serviceAccount.create`      | Specifies whether a service account should be created      | `true`                                           |
| `serviceAccount.annotations` | Annotations to add to the service account                  | `{}`                                             |
| `serviceAccount.name`        | The name of the service account to use                     | `""`                                             |
| `podAnnotations`             | Annotations to add to the pod                              | `{}`                                             |
| `podSecurityContext`         | Pod-level security attributes                              | `fsGroup: 1001`                                  |
| `securityContext`            | Container security attributes                              | `runAsGroup: 1001`<br>`runAsUser: 1001`          |
| `service.type`               | Service type (ClusterIP, NodePort or LoadBalancer)         | `ClusterIP`                                      |
| `service.port`               | Ports exposed by service                                   | `80`                                             |
| `ingress.enabled`            | Enables Ingress                                            | `false`                                          |
| `ingress.annotations`        | Ingress annotations                                        | `{}`                                             |
| `ingress.hosts.host`         | Ingress accepted hostnames                                 | `watsor.local`                                   |
| `ingress.hosts.paths`        | Ingress paths                                              | `[]`                                             |
| `ingress.tls`                | Ingress TLS configuration                                  | `[]`                                             |
| `shmSize`                    | The size of the `/dev/shm` partition for the container.    | `512Mi`                                          |
| `resources`                  | Pod resource requests & limits                             | `{}`                                             |
| `extraEnv`                   | List of environmental variables                            | `[]`                                             |
| `hostPaths`                  | List of volume mounts from the host node's filesystem      | `[]`                                             |
| `nodeSelector`               | Selector to constrain pods to nodes with particular labels | `{}`                                             |
| `tolerations`                | Tolerations to prevent scheduling onto inappropriate nodes | `[]`                                             |
| `affinity`                   | Affinity or Anti-Affinity constraints                      | `{}`                                             |
| `existingSecret`             | Name of the existing k8s secret with `secrets.yaml`        |                                                  |
| `config`                     | Watsor [configuration](https://github.com/asmirnou/watsor#configuration) as YAML block | Sample camera with color bars pattern |
